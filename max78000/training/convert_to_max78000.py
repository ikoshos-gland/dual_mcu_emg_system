"""
Model Conversion Script for MAX78000
Converts PyTorch models to MAX78000 format using ai8x-synthesis
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional

from model_architecture import create_model
from dataset import EMGDataProcessor


class MAX78000Converter:
    """
    Converter for MAX78000 deployment
    """
    
    def __init__(self, 
                 model_path: str,
                 output_dir: str,
                 preprocessor_path: Optional[str] = None):
        """
        Initialize converter
        
        Args:
            model_path: Path to trained PyTorch model
            output_dir: Output directory for converted files
            preprocessor_path: Path to preprocessor pickle file
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.preprocessor_path = preprocessor_path
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        self.model = None
        self.checkpoint = None
        self.preprocessor = None
        
        self._load_model()
        self._load_preprocessor()
    
    def _load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}")
        
        # Load checkpoint
        self.checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Determine model type from checkpoint or filename
        if 'lightweight' in self.model_path:
            model_type = 'lightweight'
        elif 'quantized' in self.model_path:
            model_type = 'quantized'
        else:
            model_type = 'standard'
        
        # Create model
        self.model = create_model(model_type)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded: {model_type}")
        print(f"Best validation accuracy: {self.checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    
    def _load_preprocessor(self):
        """Load preprocessor if available"""
        if self.preprocessor_path and os.path.exists(self.preprocessor_path):
            self.preprocessor = EMGDataProcessor()
            self.preprocessor.load_preprocessor(self.preprocessor_path)
            print(f"Preprocessor loaded from {self.preprocessor_path}")
    
    def quantize_model(self, 
                      calibration_data: Optional[np.ndarray] = None,
                      num_calibration_samples: int = 100) -> nn.Module:
        """
        Quantize model for MAX78000
        
        Args:
            calibration_data: Calibration data for quantization
            num_calibration_samples: Number of calibration samples
            
        Returns:
            Quantized model
        """
        print("Quantizing model for MAX78000...")
        
        # Prepare calibration data
        if calibration_data is None:
            # Generate synthetic calibration data
            calibration_data = np.random.randn(num_calibration_samples, 72).astype(np.float32)
        
        # Apply preprocessing if available
        if self.preprocessor and self.preprocessor.feature_scaler:
            calibration_data = self.preprocessor.feature_scaler.transform(calibration_data)
        
        # Convert to tensor
        calibration_tensor = torch.tensor(calibration_data, dtype=torch.float32)
        
        # Quantization configuration
        self.model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        
        # Prepare for quantization
        model_prepared = torch.quantization.prepare(self.model, inplace=False)
        
        # Calibrate with sample data
        with torch.no_grad():
            for i in range(0, len(calibration_tensor), 32):
                batch = calibration_tensor[i:i+32]
                model_prepared(batch)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared, inplace=False)
        
        print("Model quantization completed")
        return quantized_model
    
    def export_onnx(self, 
                   model: nn.Module,
                   filename: str = "emg_model.onnx") -> str:
        """
        Export model to ONNX format
        
        Args:
            model: PyTorch model
            filename: Output filename
            
        Returns:
            Path to exported ONNX file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Create dummy input
        dummy_input = torch.randn(1, 72)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['features'],
            output_names=['logits'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        
        print(f"ONNX model exported to {output_path}")
        return output_path
    
    def generate_ai8x_yaml(self) -> str:
        """
        Generate ai8x-synthesis YAML configuration
        
        Returns:
            Path to generated YAML file
        """
        yaml_path = os.path.join(self.output_dir, "emg_model.yaml")
        
        # ai8x-synthesis configuration
        config = {
            'arch': 'ai85net5',
            'dataset': 'emg_gestures',
            'num_classes': 8,
            'input_size': [1, 72],
            'output_size': [8],
            'layers': [
                {
                    'name': 'conv1',
                    'type': 'conv2d',
                    'kernel_size': [3, 3],
                    'stride': 1,
                    'padding': 1,
                    'in_channels': 1,
                    'out_channels': 16,
                    'activation': 'relu',
                    'batch_norm': True
                },
                {
                    'name': 'pool1',
                    'type': 'maxpool2d',
                    'kernel_size': [2, 2],
                    'stride': 2
                },
                {
                    'name': 'conv2',
                    'type': 'conv2d',
                    'kernel_size': [3, 3],
                    'stride': 1,
                    'padding': 1,
                    'in_channels': 16,
                    'out_channels': 32,
                    'activation': 'relu',
                    'batch_norm': True
                },
                {
                    'name': 'pool2',
                    'type': 'maxpool2d',
                    'kernel_size': [2, 2],
                    'stride': 2
                },
                {
                    'name': 'conv3',
                    'type': 'conv2d',
                    'kernel_size': [3, 2],
                    'stride': 1,
                    'padding': 0,
                    'in_channels': 32,
                    'out_channels': 64,
                    'activation': 'relu',
                    'batch_norm': True
                },
                {
                    'name': 'gap',
                    'type': 'avgpool2d',
                    'kernel_size': [1, 1],
                    'adaptive': True
                },
                {
                    'name': 'fc1',
                    'type': 'linear',
                    'in_features': 64,
                    'out_features': 32,
                    'activation': 'relu',
                    'dropout': 0.3
                },
                {
                    'name': 'fc2',
                    'type': 'linear',
                    'in_features': 32,
                    'out_features': 16,
                    'activation': 'relu',
                    'dropout': 0.3
                },
                {
                    'name': 'fc3',
                    'type': 'linear',
                    'in_features': 16,
                    'out_features': 8,
                    'activation': 'none'
                }
            ]
        }
        
        # Write YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"AI8X YAML config generated: {yaml_path}")
        return yaml_path
    
    def generate_weight_header(self, 
                             model: nn.Module,
                             filename: str = "emg_weights.h") -> str:
        """
        Generate C header file with weights
        
        Args:
            model: PyTorch model
            filename: Output filename
            
        Returns:
            Path to generated header file
        """
        header_path = os.path.join(self.output_dir, filename)
        
        with open(header_path, 'w') as f:
            f.write(\"\"\"/**\n * @file emg_weights.h\n * @brief Generated weights for EMG classification model\n * @author AI8X Synthesis\n */\n\n#ifndef EMG_WEIGHTS_H\n#define EMG_WEIGHTS_H\n\n#include <stdint.h>\n\n\"\"\")\n            \n            # Extract weights from model\n            layer_idx = 0\n            for name, param in model.named_parameters():\n                if param.requires_grad:\n                    weight_data = param.data.cpu().numpy()\n                    \n                    # Quantize to 8-bit\n                    weight_min = weight_data.min()\n                    weight_max = weight_data.max()\n                    \n                    if weight_max != weight_min:\n                        weight_quantized = ((weight_data - weight_min) / (weight_max - weight_min) * 255).astype(np.uint8)\n                    else:\n                        weight_quantized = np.zeros_like(weight_data, dtype=np.uint8)\n                    \n                    # Write weight array\n                    clean_name = name.replace('.', '_').replace('-', '_')\n                    f.write(f\"// Layer {layer_idx}: {name}\\n\")\n                    f.write(f\"const uint8_t {clean_name}[{weight_quantized.size}] = {{\\n\")\n                    \n                    # Write weight values\n                    for i, val in enumerate(weight_quantized.flatten()):\n                        if i % 16 == 0:\n                            f.write(\"    \")\n                        f.write(f\"0x{val:02x}\")\n                        if i < weight_quantized.size - 1:\n                            f.write(\", \")\n                        if (i + 1) % 16 == 0:\n                            f.write(\"\\n\")\n                    \n                    if weight_quantized.size % 16 != 0:\n                        f.write(\"\\n\")\n                    f.write(\"};\\n\\n\")\n                    \n                    # Write scale and zero point\n                    scale = (weight_max - weight_min) / 255.0 if weight_max != weight_min else 1.0\n                    zero_point = weight_min\n                    f.write(f\"const float {clean_name}_scale = {scale:.8f}f;\\n\")\n                    f.write(f\"const float {clean_name}_zero_point = {zero_point:.8f}f;\\n\\n\")\n                    \n                    layer_idx += 1\n            \n            f.write(\"#endif /* EMG_WEIGHTS_H */\\n\")\n        \n        print(f\"Weight header generated: {header_path}\")\n        return header_path\n    \n    def generate_inference_code(self, filename: str = \"emg_inference.c\") -> str:\n        \"\"\"Generate C inference code\"\"\"\n        code_path = os.path.join(self.output_dir, filename)\n        \n        with open(code_path, 'w') as f:\n            f.write(\"\"\"/**\n * @file emg_inference.c\n * @brief Generated inference code for EMG classification\n * @author AI8X Synthesis\n */\n\n#include <stdint.h>\n#include <string.h>\n#include \"mxc_device.h\"\n#include \"cnn.h\"\n#include \"emg_weights.h\"\n\n// Model configuration\n#define EMG_MODEL_INPUT_SIZE    72\n#define EMG_MODEL_OUTPUT_SIZE   8\n#define EMG_MODEL_LAYERS        9\n\n// Class names\nconst char* emg_class_names[EMG_MODEL_OUTPUT_SIZE] = {\n    \"Rest\",\n    \"Grasp\",\n    \"Release\",\n    \"Rotate CW\",\n    \"Rotate CCW\",\n    \"Flex\",\n    \"Extend\",\n    \"Point\"\n};\n\n/**\n * @brief Initialize CNN for EMG inference\n * @retval 0 on success\n */\nint emg_cnn_init(void)\n{\n    // Initialize CNN accelerator\n    MXC_CNN_Init();\n    \n    // Load weights (this would be generated by ai8x-synthesis)\n    // MXC_CNN_LoadWeights();\n    \n    return 0;\n}\n\n/**\n * @brief Run EMG inference\n * @param features: Input features (72 values)\n * @param output: Output class scores (8 values)\n * @retval 0 on success\n */\nint emg_cnn_infer(const int8_t* features, int32_t* output)\n{\n    // Load input data\n    // MXC_CNN_LoadInput(features);\n    \n    // Start inference\n    // MXC_CNN_Start();\n    \n    // Wait for completion\n    // while (MXC_CNN_CheckComplete() == CNN_BUSY);\n    \n    // Get results\n    // MXC_CNN_GetOutput(output);\n    \n    return 0;\n}\n\n/**\n * @brief Get predicted class\n * @param output: Class scores\n * @retval Predicted class index\n */\nint emg_get_predicted_class(const int32_t* output)\n{\n    int max_idx = 0;\n    int32_t max_val = output[0];\n    \n    for (int i = 1; i < EMG_MODEL_OUTPUT_SIZE; i++) {\n        if (output[i] > max_val) {\n            max_val = output[i];\n            max_idx = i;\n        }\n    }\n    \n    return max_idx;\n}\n\n/**\n * @brief Get class name\n * @param class_idx: Class index\n * @retval Class name string\n */\nconst char* emg_get_class_name(int class_idx)\n{\n    if (class_idx >= 0 && class_idx < EMG_MODEL_OUTPUT_SIZE) {\n        return emg_class_names[class_idx];\n    }\n    return \"Unknown\";\n}\n\"\"\")\n        \n        print(f\"Inference code generated: {code_path}\")\n        return code_path\n    \n    def generate_makefile_rules(self, filename: str = \"ai8x_rules.mk\") -> str:\n        \"\"\"Generate Makefile rules for ai8x-synthesis\"\"\"\n        makefile_path = os.path.join(self.output_dir, filename)\n        \n        with open(makefile_path, 'w') as f:\n            f.write(\"\"\"# AI8X Synthesis Rules for EMG Classification\n# Generated conversion rules\n\n# Paths\nAI8X_SYNTHESIS_PATH := $(realpath ../ai8x-synthesis)\nPYTORCH_MODEL_PATH := $(realpath ../checkpoints/best_model.pth)\nOUTPUT_DIR := $(realpath ./generated)\n\n# Generate CNN model\ngenerate-cnn-model: $(OUTPUT_DIR)/emg_weights.h $(OUTPUT_DIR)/emg_cnn_generated.h\n\n$(OUTPUT_DIR)/emg_weights.h: $(PYTORCH_MODEL_PATH)\n\t@echo \"Generating CNN weights...\"\n\t@mkdir -p $(OUTPUT_DIR)\n\t@cd $(AI8X_SYNTHESIS_PATH) && python ai8x_synthesize.py \\\n\t\t--model $(PYTORCH_MODEL_PATH) \\\n\t\t--config-file networks/emg-net.yaml \\\n\t\t--prefix emg \\\n\t\t--checkpoint-file $(PYTORCH_MODEL_PATH) \\\n\t\t--output-dir $(OUTPUT_DIR) \\\n\t\t--device MAX78000 \\\n\t\t--compact-data \\\n\t\t--mexpress \\\n\t\t--timer 0 \\\n\t\t--display-checkpoint\n\n$(OUTPUT_DIR)/emg_cnn_generated.h: $(OUTPUT_DIR)/emg_weights.h\n\t@echo \"CNN model generation complete\"\n\n# Convert PyTorch model to ONNX\ngenerate-onnx: $(OUTPUT_DIR)/emg_model.onnx\n\n$(OUTPUT_DIR)/emg_model.onnx: $(PYTORCH_MODEL_PATH)\n\t@echo \"Converting to ONNX...\"\n\t@mkdir -p $(OUTPUT_DIR)\n\t@python convert_to_max78000.py \\\n\t\t--model $(PYTORCH_MODEL_PATH) \\\n\t\t--output-dir $(OUTPUT_DIR) \\\n\t\t--format onnx\n\n# Clean generated files\nclean-generated:\n\t@echo \"Cleaning generated files...\"\n\t@rm -rf $(OUTPUT_DIR)\n\n.PHONY: generate-cnn-model generate-onnx clean-generated\n\"\"\")\n        \n        print(f\"Makefile rules generated: {makefile_path}\")\n        return makefile_path\n    \n    def convert_model(self, export_formats: list = ['onnx', 'weights', 'inference']) -> Dict[str, str]:\n        \"\"\"Convert model to MAX78000 format\"\"\"\n        print(\"Starting model conversion for MAX78000...\")\n        \n        results = {}\n        \n        # Quantize model\n        quantized_model = self.quantize_model()\n        \n        # Export in requested formats\n        if 'onnx' in export_formats:\n            results['onnx'] = self.export_onnx(quantized_model)\n        \n        if 'weights' in export_formats:\n            results['weights'] = self.generate_weight_header(quantized_model)\n        \n        if 'inference' in export_formats:\n            results['inference'] = self.generate_inference_code()\n        \n        if 'makefile' in export_formats:\n            results['makefile'] = self.generate_makefile_rules()\n        \n        # Generate configuration info\n        config_info = {\n            'model_type': self.model.__class__.__name__,\n            'input_size': 72,\n            'output_size': 8,\n            'num_parameters': sum(p.numel() for p in self.model.parameters()),\n            'best_accuracy': self.checkpoint.get('best_val_acc', 0.0),\n            'conversion_timestamp': str(torch.datetime.now()),\n            'export_formats': export_formats\n        }\n        \n        config_path = os.path.join(self.output_dir, 'conversion_info.json')\n        with open(config_path, 'w') as f:\n            json.dump(config_info, f, indent=2)\n        \n        results['config'] = config_path\n        \n        print(\"Model conversion completed!\")\n        print(f\"Output directory: {self.output_dir}\")\n        for format_type, path in results.items():\n            print(f\"  {format_type}: {path}\")\n        \n        return results\n\n\ndef main():\n    \"\"\"Main conversion function\"\"\"\n    parser = argparse.ArgumentParser(description='Convert PyTorch model to MAX78000')\n    parser.add_argument('--model', type=str, required=True,\n                       help='Path to trained PyTorch model')\n    parser.add_argument('--output-dir', type=str, default='./generated',\n                       help='Output directory for converted files')\n    parser.add_argument('--preprocessor', type=str, default=None,\n                       help='Path to preprocessor pickle file')\n    parser.add_argument('--formats', nargs='+', \n                       default=['onnx', 'weights', 'inference', 'makefile'],\n                       choices=['onnx', 'weights', 'inference', 'makefile'],\n                       help='Export formats')\n    \n    args = parser.parse_args()\n    \n    # Create converter\n    converter = MAX78000Converter(\n        model_path=args.model,\n        output_dir=args.output_dir,\n        preprocessor_path=args.preprocessor\n    )\n    \n    # Convert model\n    results = converter.convert_model(export_formats=args.formats)\n    \n    print(\"\\nConversion Summary:\")\n    print(\"=\" * 50)\n    for format_type, path in results.items():\n        print(f\"{format_type.upper()}: {path}\")\n\n\nif __name__ == \"__main__\":\n    main()