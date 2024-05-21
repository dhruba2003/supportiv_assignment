# supportiv_assignment

This project implements a sequence-to-sequence (seq2seq) model for a question-answering system using a transformer-based architecture, specifically the T5 model from Hugging Face's Transformers library. The model is trained on a dataset of question-answer pairs and can generate answers to user queries.

## Approach

### Data pre-processing

The dataset is assumed to be in the form of a csv file containing two columns 'question' and 'answer'. If there are duplicate rows of questions it is assumed that only the first instance of the row is true and other duplicate instances are removed. The dataset is then separated into two lists one containing the 'questions' and one containing the 'columns'. Next step is the removal of punctuations and tokenizing the data. The entire data is tokenized using a tokenizer of 't-5' and labels are also added and 'token-input-ids' are removed.

### Model Selection

We use the T5 model (t5-small) from Hugging Face's Transformers library. T5 is a versatile transformer model that can be used for various text generation tasks, including question-answering.

### Training

The model is fine-tuned using the Seq2SeqTrainer class provided by the Transformers library. The training process involves the following steps:

-Tokenizing the input questions and target answers.

-Preparing a custom dataset class to handle the tokenized data.

-Setting up training arguments, such as batch size, learning rate, and number of epochs.

-Training the model on the tokenized dataset.

### Inference

After training, the model is saved and loaded for inference. A text generation pipeline is created to generate answers to user queries based on the trained model.

### Assumptions

-Tokenization: The tokenization process adequately handles the maximum sequence length, and the input and output sequences are appropriately padded.

-Model Configuration: The T5 model configuration (t5-small) is sufficient for the question-answering task at hand.

### Model Performance

Strengths

-Versatility: The T5 model is capable of handling various text generation tasks, making it a powerful tool for question-answering.

-Pre-trained Knowledge: Leveraging a pre-trained transformer model like T5 provides a strong foundation of language understanding, leading to better performance on the task.

Weaknesses

-Resource Intensive: Transformer models require significant computational resources for training and inference, which can be a limitation for large-scale applications.

-Data Dependency: The quality and size of the training dataset directly impact the model's performance. Limited or low-quality data can lead to suboptimal results.

### Evaluation Metrics

-The general loss function used here is privy to the huggingface transformers. The loss function usesd here is the cross-entropy loss which is suitable for sequence to sequence tasks.

-The other two metrics used were the BLEU score and the rogue score. These two metrics were selected as they are the standard metrics used in text generation and text summarization tasks respectively. These metrics provide insights into how well the generated answers match the reference answers, helping to identify areas for improvement.

### Potential Improvements

-Data Augmentation: Enhancing the dataset with more question-answer pairs or using data augmentation techniques can improve the model's performance.

-Model Fine-tuning: Experimenting with different transformer architectures (e.g., BERT, GPT-3) or larger model sizes (e.g., t5-base, t5-large) might yield better results.

-Hyperparameter Tuning: Conducting a thorough hyperparameter search to optimize learning rates, batch sizes, and other training parameters can lead to better model performance.

-Attention Mechanisms: Incorporating advanced attention mechanisms or using models with built-in attention layers can improve the model's ability to generate accurate answers.
