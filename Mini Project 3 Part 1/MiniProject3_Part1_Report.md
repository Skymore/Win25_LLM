## Fine-tuning LLaMA 3.2-1B for Buyer Intent Classification

### Group Name: 1(Group 18)

### Team Members and Contributions

#### Victoria CHENG
- Implemented zero-shot and few-shot experiments
- Set up the model access and data loading

#### Rui TAO
- Designed and implemented LoRA fine-tuning experiments
  - Created prompt templates for different evaluation approaches
  - Optimized training parameters and evaluation strategies


### Model Performance Comparison
- LLaMA 3.2-1B showed mixed zero-shot performance (Macro F1=0.5509) but after fine-tuning improved dramatically (Macro F1=0.8689).
- Previously challenging categories like "Offensive Intent" improved from F1=0.0548 to F1=0.8430 after fine-tuning.

### LoRA Configuration & Hyperparameters
- Higher rank LoRA configuration (r=16, alpha=32) targeting attention layers significantly outperformed lower rank settings.
- Targeting attention mechanism layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) provided better capacity for learning intent classification.

### Hyperparameter Tuning & Training Strategy
- Batch size 4 with gradient accumulation steps of 8 and moderate learning rate (3e-5) provided optimal training stability.
- Three epochs were sufficient to avoid overfitting, with evaluation every 200 steps.
- The direct prompt template consistently outperformed step-by-step reasoning across all configurations.

### Future Improvements & Lessons Learned
- Expanding the training dataset with more diverse examples and implementing specialized loss functions for class imbalance could further improve performance.
- LoRA fine-tuning achieved excellent results while training only 0.5% of parameters, making efficient adaptation of LLMs practical even with limited computational resources.
