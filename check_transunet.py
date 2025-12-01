"""
Diagnostic script to check TransUNet parameters
"""

import inspect
from transunet import TransUNet

# Get the __init__ signature
sig = inspect.signature(TransUNet.__init__)
print("\n" + "="*60)
print("TransUNet.__init__() Parameters:")
print("="*60)
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
        print(f"  {param_name}: {default}")

print("\n" + "="*60)
print("How to create your TransUNet model:")
print("="*60)

# Try to instantiate with minimal parameters
try:
    model = TransUNet()
    print("✓ TransUNet() works with no parameters")
except TypeError as e:
    print(f"✗ TransUNet() requires parameters: {e}")

# Common parameter patterns
test_configs = [
    {'img_dim': 224, 'num_classes': 14},
    {'img_size': 224, 'num_classes': 14},
    {'img_dim': 224, 'n_classes': 14},
    {'image_size': 224, 'num_classes': 14},
]

for config in test_configs:
    try:
        model = TransUNet(**config)
        print(f"✓ TransUNet({config}) WORKS!")
        break
    except TypeError:
        print(f"✗ TransUNet({config}) failed")

print("="*60)
