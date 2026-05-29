"""LLM Service for answer generation using HuggingFace models"""

import os
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)


class LLMService:
    """Service for generating answers using HuggingFace LLMs"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        use_quantization: bool = True,
        device: Optional[str] = None,
        hf_token: Optional[str] = None
    ):
        """
        Initialize the LLM service.
        
        Args:
            model_name: HuggingFace model name or path
            use_quantization: Use 4-bit quantization to save memory (recommended for Colab)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            hf_token: Optional Hugging Face token for gated/private models
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.device = device or self._detect_device()
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        self.tokenizer = None
        self.model = None
        self.generator = None
        self._load_model()

    def _detect_device(self) -> str:
        """Choose the best locally available inference device."""
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self):
        """Load the LLM model and tokenizer"""
        print(f"Loading LLM model: {self.model_name}")
        print(f"Device: {self.device}")

        try:
            self._load_tokenizer_and_model()
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM model {self.model_name}: {e}") from e

    def _load_tokenizer_and_model(self):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=self.hf_token
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization for memory efficiency (useful for Colab)
        uses_device_map = False
        if self.use_quantization and self.device == "cuda":
            print("Using 4-bit quantization to save memory...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            uses_device_map = True
        else:
            quantization_config = None

        # Load model
        if quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                token=self.hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            uses_device_map = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=self.hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            if self.device in {"cpu", "mps"}:
                self.model = self.model.to(self.device)
            uses_device_map = False

        # Store flag for pipeline creation
        self.uses_device_map = uses_device_map

        # Create pipeline for text generation.
        # When using device_map="auto", don't specify device in pipeline.
        pipeline_device = -1
        if self.device == "cuda":
            pipeline_device = 0
        elif self.device == "mps":
            pipeline_device = torch.device("mps")

        self.generator = (
            pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
            if self.uses_device_map
            else pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=pipeline_device,
            )
        )
        print("LLM model loaded successfully")
    
    def generate_answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9
    ) -> str:
        """
        Generate an answer based on the question and retrieved context.
        
        Args:
            question: The user's question
            context: Retrieved context from documents
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated answer text
        """
        # Create prompt based on model type
        if "instruct" in self.model_name.lower() or "chat" in self.model_name.lower():
            # For instruction-tuned models (Mistral, Llama-2-chat, etc.)
            prompt = self._create_instruction_prompt(question, context)
        else:
            # For base models
            prompt = self._create_base_prompt(question, context)
        
        try:
            # Generate response
            do_sample = temperature > 0
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "num_return_sequences": 1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "truncation": True,
                "return_full_text": False,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p

            outputs = self.generator(prompt, **generation_kwargs)
            
            return outputs[0]['generated_text'].strip()
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _create_instruction_prompt(self, question: str, context: str) -> str:
        """Create a prompt for instruction-tuned models"""
        # Try to use chat template if available (for models like Mistral, Llama-2-chat)
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": f"""You are a careful document question-answering assistant. Answer only from the context.
If the context contains the answer, give the direct answer in one or two sentences.
If the context does not contain the answer, say that the context does not provide enough information.

Context:
{context}

Question: {question}"""
                    }
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception:
                pass  # Fall back to simple prompt
        
        # Simple prompt format (works for most instruction models)
        prompt = f"""You are a careful document question-answering assistant. Answer only from the context.
If the context contains the answer, give the direct answer in one or two sentences.
If the context does not contain the answer, say that the context does not provide enough information.

Context:
{context}

Question: {question}

Answer:"""
        return prompt
    
    def _create_base_prompt(self, question: str, context: str) -> str:
        """Create a prompt for base models"""
        prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {question}

Answer:"""
        return prompt


class SimpleLLMService:
    """Simpler LLM service using smaller, faster models (for CPU/quick testing)"""
    
    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize with a smaller model (faster, less memory).
        
        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.generator = None
        self._load_model()
    
    def _load_model(self):
        """Load a simple model"""
        print(f"Loading simple LLM: {self.model_name}")
        try:
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                device=-1  # CPU
            )
            print("✓ Simple LLM loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def generate_answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        top_p: float = 0.9
    ) -> str:
        """Generate a simple answer"""
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": 1,
            "return_full_text": False,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        outputs = self.generator(prompt, **generation_kwargs)
        return outputs[0]['generated_text'].strip()
