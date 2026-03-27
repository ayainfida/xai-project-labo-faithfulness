import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from models.asso_opt.asso_opt import AssoConceptFast
import clip
import json
import os
import pickle
import argparse
import random

device = "cuda" if th.cuda.is_available() else "cpu"

# ── load CLIP once ───────────────────────────────────────────────
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()

# ── load LLaVA once ──────────────────────────────────────────────
llava_proc  = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
llava_model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=th.float16, device_map="auto")
llava_model.eval()

# ── token ids — derived from tokenizer, not hardcoded ────────────
YES_ID = llava_proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
NO_ID  = llava_proc.tokenizer.encode("No",  add_special_tokens=False)[0]
print(f"YES_ID={YES_ID}, NO_ID={NO_ID}")

def safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0


def resolve_image_path(image_dir, fname):
    candidate = os.path.join(image_dir, fname)
    if os.path.exists(candidate):
        return candidate

    root, ext = os.path.splitext(fname)
    if ext:
        return None

    for extension in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = os.path.join(image_dir, f"{fname}{extension}")
        if os.path.exists(candidate):
            return candidate
    return None

def llava_score(image, concept):
    conv = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text",
         "text": f"Is the following property clearly visible in this image? "
                 f"Answer only Yes or No.\nProperty: {concept}"}
    ]}]
    prompt = llava_proc.apply_chat_template(conv, add_generation_prompt=True)
    inputs = llava_proc(image, prompt, return_tensors="pt").to(llava_model.device, th.float16)
    with th.inference_mode():
        outputs = llava_model(**inputs)
    logits = outputs.logits[0, -1]
    score  = th.softmax(logits[[YES_ID, NO_ID]], dim=0)[0].item()
    margin = (logits[YES_ID] - logits[NO_ID]).item()
    return score, margin


def run_faithfulness_audit(ckpt, split_file, image_dir, save_path,
                            n_per_class=2, top_k=5, seed=42):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    # ── load model ───────────────────────────────────────────────
    model = AssoConceptFast.load_from_checkpoint(ckpt, weights_only=False)
    model.eval().to(device)
    concept_texts = model.concept_raw
    concept_feat  = model.concepts.float()
    concept_feat  = concept_feat / concept_feat.norm(dim=-1, keepdim=True)
    asso_softmax  = F.softmax(model.asso_mat, dim=-1).detach()
    print(f"Loaded: {len(concept_texts)} concepts | {asso_softmax.shape[0]} classes")

    # ── load split ───────────────────────────────────────────────
    with open(split_file, "rb") as f:
        class2images = pickle.load(f)
    class_names = sorted(class2images.keys())
    print(f"Classes: {len(class_names)}")

    # ── stratified random sampling with seed ─────────────────────
    random.seed(seed)
    np.random.seed(seed)
    image_paths, image_labels = [], []
    skipped_missing = 0
    for class_idx, class_name in enumerate(class_names):
        imgs = class2images[class_name]
        # randomly sample n_per_class — reproducible with seed
        sampled = random.sample(imgs, min(n_per_class, len(imgs)))
        for fname in sampled:
            resolved_path = resolve_image_path(image_dir, fname)
            if resolved_path is None:
                skipped_missing += 1
                print(f"Skipping {os.path.join(image_dir, fname)}: file not found (tried common extensions)")
                continue
            image_paths.append(resolved_path)
            image_labels.append(class_idx)

    print(f"Total images: {len(image_paths)} ({n_per_class} per class, seed={seed})")
    if skipped_missing:
        print(f"Skipped missing files during sampling: {skipped_missing}")

    # ── resume if interrupted ────────────────────────────────────
    if os.path.exists(save_path):
        with open(save_path) as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"Resuming from {start_idx}")
    else:
        results = []
        start_idx = 0

    # ── main loop ────────────────────────────────────────────────
    for i in range(start_idx, len(image_paths)):
        img_path = image_paths[i]
        true_cls = image_labels[i]

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Skipping {img_path}: {e}")
            continue

        with th.no_grad():
            feat         = clip_model.encode_image(
                               preprocess(pil_img).unsqueeze(0).to(device)).float()
            feat         = feat / feat.norm(dim=-1, keepdim=True)
            dot_product  = feat @ concept_feat.T              # (1, C)
            class_scores = model.forward(dot_product) * 100   # (1, num_classes)
            pred_cls     = int(class_scores.argmax(dim=-1).item())
            activation   = dot_product.squeeze(0)             # (C,)
            contrib      = activation * asso_softmax[pred_cls] # (C,)
            top_ids      = th.topk(contrib, top_k).indices.cpu().numpy()
            avg_clip_topk = float(activation[top_ids].mean().item())

        yes_count = 0
        top_info  = []
        for idx in top_ids:
            concept_str = str(concept_texts[idx])
            score, margin       = llava_score(pil_img, concept_str)
            visible     = score > 0.5
            yes_count  += int(visible)
            top_info.append({
                "concept":     concept_str,
                "clip_score":  round(float(activation[idx].item()), 4),
                "llava_p_yes": round(score, 3),
                "llava_margin":  round(float(margin), 3),
                "visible":     visible
            })

        faithfulness = yes_count / top_k
        results.append({
            "img":          os.path.basename(img_path),
            "true_class":   class_names[true_cls],
            "predicted":    class_names[pred_cls],
            "correct":      pred_cls == true_cls,
            "faithfulness": faithfulness,
            "avg_clip_topk": round(avg_clip_topk, 4),
            "top5":         top_info
        })

        # print per image with concepts
        print(f"\n[{i+1:4d}/{len(image_paths)}] {os.path.basename(img_path)}")
        print(f"  true={class_names[true_cls]:25s} pred={class_names[pred_cls]:25s} "
              f"correct={pred_cls==true_cls} faith={faithfulness:.2f}")
        print(f"  Top concepts:")
        for c in top_info:
            tick = "✓" if c["visible"] else "✗"
            print(f"    {tick} p={c['llava_p_yes']:.3f} m={c['llava_margin']:.3f} | {c['concept'][:60]}")

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

    # ── summary ──────────────────────────────────────────────────
    faith_scores    = [r["faithfulness"] for r in results]
    correct_faith   = [r["faithfulness"] for r in results if r["correct"]]
    incorrect_faith = [r["faithfulness"] for r in results if not r["correct"]]

    summary = {
        "dataset":         save_path,
        "n_images":        len(results),
        "n_correct":       sum(r["correct"] for r in results),
        "accuracy":        round(sum(r["correct"] for r in results) / len(results), 3) if len(results) > 0 else 0.0,
        "mean_faith":      round(safe_mean(faith_scores), 3),
        "correct_faith":   round(safe_mean(correct_faith), 3),
        "incorrect_faith": round(safe_mean(incorrect_faith), 3),
        "low_faith_count": sum(s < 0.4 for s in faith_scores),
        "low_faith_pct":   round(sum(s < 0.4 for s in faith_scores) / len(results), 3) if len(results) > 0 else 0.0,
    }

    print(f"\n{'='*55}")
    print(f"Dataset:                      {save_path}")
    print(f"Images evaluated:             {summary['n_images']}")
    print(f"Accuracy:                     {summary['accuracy']:.3f}")
    print(f"Mean faithfulness:            {summary['mean_faith']:.3f}")
    print(f"Faithfulness when correct:    {summary['correct_faith']:.3f}")
    print(f"Faithfulness when incorrect:  {summary['incorrect_faith']:.3f}")
    print(f"Images with faith < 0.4:      {summary['low_faith_count']} "
          f"({summary['low_faith_pct']*100:.1f}%)")

    # save summary separately
    summary_path = save_path.replace(".json", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True)
    parser.add_argument("--split_file",  required=True)
    parser.add_argument("--image_dir",   required=True)
    parser.add_argument("--save_path",   required=True)
    parser.add_argument("--n_per_class", type=int, default=2)
    parser.add_argument("--top_k",       type=int, default=5)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    run_faithfulness_audit(
        ckpt=args.ckpt,
        split_file=args.split_file,
        image_dir=args.image_dir,
        save_path=args.save_path,
        n_per_class=args.n_per_class,
        top_k=args.top_k,
        seed=args.seed
    )