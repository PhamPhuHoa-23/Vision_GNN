# Enhanced ViGNN-CLIP Training

Phiên bản cải tiến của ViGNN-CLIP với **Sentence-BERT** cho text embedding và **Pyramid ViG-S** cho image embedding.

## 🌟 Tính năng mới

### 🔧 **Cải tiến chính:**
- **Sentence-BERT**: Thay thế simple text encoder bằng pre-trained Sentence-BERT models
- **Enhanced Pyramid ViG**: Cải tiến Pyramid ViG với graph attention và adaptive pooling
- **Advanced Loss Functions**: Multi-scale contrastive loss với local và global components
- **Memory Efficient**: Tối ưu hóa cho single GPU training
- **Comprehensive Evaluation**: Đánh giá toàn diện với retrieval metrics

### 📊 **Kiến trúc:**
```
Image → Enhanced PVigS → Graph Attention → Adaptive Pooling → [512-dim vector]
                                                                      ↓
                                                              Contrastive Loss
                                                                      ↑
Text → Sentence-BERT → Projection Layer → LayerNorm → [512-dim vector]
```

## 🚀 Cài đặt

### 1. **Dependencies:**
```bash
pip install -r requirements_enhanced.txt
```

### 2. **Sentence-BERT models (tự động download):**
- `all-MiniLM-L6-v2`: 384D, nhanh, hiệu quả
- `all-mpnet-base-v2`: 768D, chậm hơn nhưng tốt hơn
- `paraphrase-multilingual-MiniLM-L12-v2`: Đa ngôn ngữ

## 📂 Chuẩn bị dữ liệu

### **Supported formats:**

#### **Format 1: CSV (LAION-style)**
```
data/
├── metadata.csv          # columns: image_file, caption, url, similarity
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

#### **Format 2: JSON (COCO-style)**
```
data/
├── annotations.json      # COCO format hoặc simple [{image: "...", caption: "..."}]
└── images/
    ├── image1.jpg
    └── ...
```

#### **Format 3: Text file**
```
data/
├── train.txt            # image_path<TAB>caption
└── images/
    └── ...
```

#### **Format 4: Directory structure**
```
data/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── captions/
    ├── image1.txt       # Caption for image1.jpg
    └── image2.txt       # Caption for image2.jpg
```

## 🏋️ Training

### **Basic training:**
```bash
python vignn_clip_training_script_improved.py \
    --data_path "/path/to/dataset" \
    --model_name pvig_s_224_gelu \
    --text_model all-MiniLM-L6-v2 \
    --embedding_dim 512 \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir "./outputs"
```

### **Advanced training with pretrained weights:**
```bash
python vignn_clip_training_script_improved.py \
    --data_path "/path/to/dataset" \
    --model_name pvig_s_224_gelu \
    --text_model all-MiniLM-L6-v2 \
    --embedding_dim 512 \
    --pretrained_vignn "pretrained_weights/pvig_s_82.1.pth.tar" \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --temperature 0.07 \
    --lambda_local 0.1 \
    --lambda_global 0.05 \
    --augmentation medium \
    --scheduler cosine \
    --gradient_clip_val 1.0 \
    --output_dir "./outputs" \
    --log_wandb \
    --save_every 10
```

### **Using the training script:**
```bash
# Chỉnh sửa DATA_PATH trong file train_enhanced.sh
bash train_enhanced.sh
```

## 🔧 Tham số quan trọng

### **Model Configuration:**
- `--model_name`: `pvig_ti_224_gelu`, `pvig_s_224_gelu`, `pvig_m_224_gelu`, `pvig_b_224_gelu`
- `--text_model`: Sentence-BERT model name
- `--embedding_dim`: Dimension của output embeddings (512, 768, 1024)
- `--freeze_text_encoder`: Freeze Sentence-BERT weights

### **Training Settings:**
- `--batch_size`: Batch size (32 cho GPU 24GB)
- `--lr`: Learning rate (1e-4 recommended)
- `--weight_decay`: Weight decay (0.05)
- `--warmup_epochs`: Warmup epochs (5)
- `--gradient_clip_val`: Gradient clipping (1.0)

### **Loss Configuration:**
- `--temperature`: Temperature cho contrastive loss (0.07)
- `--lambda_local`: Weight cho local alignment loss (0.1)
- `--lambda_global`: Weight cho global consistency loss (0.05)

### **Data Augmentation:**
- `--augmentation`: `light`, `medium`, `strong`

## 📊 Evaluation & Testing

### **Load và test model:**
```bash
python test_enhanced_model.py \
    --model_path "outputs/best_model.pth" \
    --config_path "outputs/config.json" \
    --test_type retrieval \
    --image_dir "/path/to/test/images" \
    --text_file "/path/to/test/captions.txt" \
    --max_samples 1000
```

### **Image-to-Text Retrieval:**
```bash
python test_enhanced_model.py \
    --model_path "outputs/best_model.pth" \
    --test_type retrieval \
    --image_path "/path/to/image.jpg" \
    --text_file "/path/to/candidates.txt" \
    --top_k 5
```

### **Text-to-Image Retrieval:**
```bash
python test_enhanced_model.py \
    --model_path "outputs/best_model.pth" \
    --test_type retrieval \
    --query_text "A beautiful sunset over the ocean" \
    --image_dir "/path/to/images" \
    --top_k 5
```

### **Encode embeddings:**
```bash
python test_enhanced_model.py \
    --model_path "outputs/best_model.pth" \
    --test_type encode \
    --image_path "/path/to/image.jpg" \
    --query_text "Your caption here"
```

## 📈 Monitoring

### **WandB Integration:**
- Tự động log training metrics, loss components, retrieval performance
- Visualizations: loss curves, similarity matrices, embedding distributions

### **Metrics được track:**
- **Training**: total_loss, contrastive_loss, local_loss, global_loss, learning_rate
- **Validation**: I2T R@1,5,10, T2I R@1,5,10, mean_recall, RSum

## 💾 Output Structure

```
outputs/
├── config.json              # Training configuration
├── best_model.pth           # Best model checkpoint
├── checkpoint_epoch_X.pth   # Regular checkpoints
└── logs/                    # Training logs
```

### **Model checkpoint format:**
```python
{
    'image_encoder': state_dict,
    'text_encoder': state_dict,
    'optimizer': state_dict,
    'scheduler': state_dict,
    'epoch': int,
    'metrics': dict,
    'args': dict
}
```

## 🔧 Memory Management

### **GPU Memory optimization:**
- Mixed precision training (AMP)
- Gradient checkpointing
- Efficient data loading với pin_memory
- Batch size tự động điều chỉnh theo GPU memory

### **Recommended settings:**
- **GTX 1080Ti (11GB)**: batch_size=16, pvig_ti
- **RTX 3080 (10GB)**: batch_size=24, pvig_s  
- **RTX 3090 (24GB)**: batch_size=32, pvig_s/m
- **A100 (40GB)**: batch_size=64, pvig_b

## 🎯 Performance Benchmarks

### **Expected Results (COCO 5K test):**
| Model | Text Encoder | I2T R@1 | I2T R@5 | T2I R@1 | T2I R@5 | RSum |
|-------|--------------|---------|---------|---------|---------|------|
| PVigS-Ti | MiniLM-L6 | 45-50% | 75-80% | 35-40% | 65-70% | 220-240 |
| PVigS-S | MiniLM-L6 | 50-55% | 80-85% | 40-45% | 70-75% | 240-260 |
| PVigS-S | MPNet-Base | 55-60% | 85-90% | 45-50% | 75-80% | 260-280 |

## 🐛 Troubleshooting

### **Common Issues:**

#### **1. Out of Memory:**
```bash
# Giảm batch size
--batch_size 16

# Sử dụng model nhỏ hơn
--model_name pvig_ti_224_gelu

# Freeze text encoder
--freeze_text_encoder
```

#### **2. Slow training:**
```bash
# Giảm số workers
--num_workers 2

# Sử dụng text model nhỏ hơn
--text_model all-MiniLM-L6-v2
```

#### **3. Poor convergence:**
```bash
# Điều chỉnh learning rate
--lr 5e-5

# Tăng warmup
--warmup_epochs 10

# Điều chỉnh loss weights
--lambda_local 0.05 --lambda_global 0.02
```

## 📝 Tips & Best Practices

### **Training Tips:**
1. **Start small**: Test với `--max_samples 1000` trước
2. **Monitor convergence**: Kiểm tra validation metrics mỗi 5 epochs
3. **Use pretrained weights**: Luôn load pretrained ViGNN weights
4. **Batch size**: Càng lớn càng tốt (trong giới hạn memory)
5. **Learning rate**: Start với 1e-4, scale theo batch size

### **Data Preparation:**
1. **Caption quality**: Filter captions quá ngắn/dài
2. **Image resolution**: Resize về 224x224 trước training
3. **Balance dataset**: Đảm bảo diverse captions
4. **Text cleaning**: Remove HTML tags, normalize whitespace

### **Inference Optimization:**
1. **Batch inference**: Encode nhiều images/texts cùng lúc
2. **Cache embeddings**: Save embeddings to disk cho reuse
3. **GPU utilization**: Use GPU cho inference khi có thể

## 🔗 Related Work

- **Original ViG**: [Vision GNN: An Image is Worth Graph of Nodes](https://arxiv.org/abs/2206.00272)
- **Sentence-BERT**: [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- **CLIP**: [Learning Transferable Visual Representations](https://arxiv.org/abs/2103.00020)

## 📄 License

This project is based on the original ViG implementation and follows the same license terms.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📞 Support

Nếu gặp vấn đề:
1. Check Issues trên GitHub
2. Review troubleshooting section
3. Create new issue với detailed logs 