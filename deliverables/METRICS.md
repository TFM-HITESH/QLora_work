# Metrics Pipeline: Implementation Guide

## 1. Introduction
This document outlines the pipeline for collecting and evaluating key performance metrics for the QLoRA-enabled video-text models. Accurate and comprehensive metric collection is crucial for assessing model performance, memory efficiency, and training speed, as detailed in the project's `README.md` and `GOALS.md`.

## 2. Metric Categories
Based on the project requirements, metrics are categorized into three main areas:

### 2.1. Task Performance Metrics
These metrics evaluate the model's effectiveness on specific video-text tasks.

#### 2.1.1. Video Captioning
-   **BLEU (Bilingual Evaluation Understudy):** Measures the n-gram overlap between generated and reference captions. (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
-   **CIDEr (Consensus-based Image Description Evaluation):** Measures the consensus of a generated caption with a set of ground truth captions, giving higher weight to n-grams that are more frequent in the ground truth.
-   **METEOR (Metric for Evaluation of Translation with Explicit Ordering):** Considers precision, recall, and word-to-word matching, including synonyms and paraphrases.

#### 2.1.2. Video Retrieval
-   **Recall@K (R@K):** Measures the proportion of queries for which the correct item (video or text) is found within the top K retrieved results.
-   **mAP (mean Average Precision):** The mean of the Average Precision (AP) scores across all queries. AP is the area under the precision-recall curve.

### 2.2. Efficiency Metrics
These metrics quantify the computational and memory footprint of the models.

-   **Memory Usage (VRAM):**
    -   **Peak Training VRAM:** Maximum GPU memory consumed during the training process.
    -   **Inference VRAM:** GPU memory consumed during model inference.
-   **Training Time:**
    -   **Time per Epoch:** Duration to complete one full pass over the training dataset.
    -   **Total Training Time:** Overall time taken for the entire fine-tuning process.
-   **FLOPs (Floating Point Operations):** Total number of floating-point operations performed during inference or a forward pass.
-   **Latency:** Time taken for a single inference query (e.g., generating a caption for one video).

### 2.3. Ablation Study Metrics
These metrics are used to analyze the impact of different hyperparameters and architectural choices.

-   **Performance vs. Rank Size (r):** How task performance changes with varying LoRA ranks.
-   **Performance vs. Quantization Bit-width:** Comparison between 4-bit, 8-bit, and potentially full precision.
-   **Performance vs. Adapter Placement:** Impact of applying LoRA to different parts of the model (e.g., vision encoder, text decoder, cross-modal fusion).
-   **Performance vs. Frame Sampling Density:** How performance is affected by the number of frames used for video input.

## 3. Implementation Guide for Metric Collection

### 3.1. Setup and Environment
-   **Libraries:** Utilize standard libraries for metric calculation. For NLP metrics, `nltk`, `pycocoevalcap` (for CIDEr, BLEU, METEOR) are common. For general deep learning metrics, `torch.cuda.max_memory_allocated()` for VRAM, and `time` module for timing.
-   **Logging:** Integrate with experiment tracking tools like Weights & Biases, TensorBoard, or MLflow to log metrics, hyperparameters, and system resources.

### 3.2. Task Performance Metric Collection

#### 3.2.1. Video Captioning
1.  **Generate Captions:** During the evaluation phase, use the fine-tuned model to generate captions for the test set videos.
2.  **Prepare References:** Ensure you have a set of ground truth reference captions for each video in the test set.
3.  **Calculate Metrics:**
    ```python
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor

    # Example structure for results and references
    # results = {image_id: [generated_caption]}
    # references = {image_id: [ref1, ref2, ...]}

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr")
    ]

    for scorer, method in scorers:
        score, scores = scorer.compute_score(references, results)
        if type(method) == list:
            for m, s in zip(method, score):
                print(f"{m}: {s:.4f}")
        else:
            print(f"{method}: {score:.4f}")
    ```

#### 3.2.2. Video Retrieval
1.  **Generate Embeddings:** Obtain video embeddings and text query embeddings from the model for the test set.
2.  **Similarity Calculation:** Compute cosine similarity (or other appropriate similarity measure) between video and text embeddings.
3.  **Ranking:** For each query (video or text), rank the candidate items based on similarity.
4.  **Calculate R@K and mAP:**
    ```python
    # Conceptual pseudocode
    def calculate_recall_at_k(rankings, ground_truths, k):
        # Logic to check if ground truth is in top K
        pass

    def calculate_map(rankings, ground_truths):
        # Logic to calculate Average Precision for each query and then mean
        pass

    # Example usage
    # video_to_text_rankings = {video_id: [ranked_text_ids]}
    # text_to_video_rankings = {text_id: [ranked_video_ids]}
    # ground_truths = {query_id: [relevant_item_ids]}

    # R@1, R@5, R@10 for video-to-text and text-to-video retrieval
    # mAP for both directions
    ```

### 3.3. Efficiency Metric Collection

#### 3.3.1. Memory Usage (VRAM)
-   **PyTorch:**
    ```python
    import torch

    # Before training/inference
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # During training/inference
    # ... model forward/backward pass ...

    # After training/inference
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3) # in GB
    print(f"Peak VRAM: {peak_vram:.2f} GB")
    ```

#### 3.3.2. Training Time
-   Use Python's `time` module:
    ```python
    import time

    start_time = time.time()
    # ... training loop for one epoch ...
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f"Epoch duration: {epoch_duration:.2f} seconds")
    ```

#### 3.3.3. FLOPs and Latency
-   **FLOPs:** Can be estimated using libraries like `thop` (for PyTorch) or by analyzing model architecture. For more precise measurements, profiling tools can be used.
-   **Latency:** Measure the time taken for a single forward pass (inference) on a representative input.
    ```python
    import time

    # Warm-up runs
    for _ in range(10): model(dummy_input)

    start_time = time.time()
    model(actual_input)
    end_time = time.time()
    latency = (end_time - start_time) * 1000 # in milliseconds
    print(f"Inference Latency: {latency:.2f} ms")
    ```

## 4. Reporting and Visualization
-   **Tables:** Present quantitative results in clear, well-formatted tables comparing different models and configurations.
-   **Graphs:** Visualize trends for ablation studies (e.g., performance vs. rank size, memory vs. performance Pareto curves).
-   **Experiment Tracking Platforms:** Utilize tools like Weights & Biases or TensorBoard to automatically log, visualize, and compare experiment runs.

## 5. Conclusion
Implementing a robust metrics pipeline is essential for systematically evaluating the effectiveness and efficiency of QLoRA in video-text models. By collecting and analyzing these metrics, we can gain insights into the trade-offs between performance, memory usage, and computational cost, guiding further optimization and research efforts.
