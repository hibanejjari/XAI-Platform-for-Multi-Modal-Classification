import torchxrayvision as xrv

print("Loading model + weights...")
model = xrv.models.DenseNet(weights="densenet121-res224-chex")
print("Loaded. Pathologies:", model.pathologies[:5], "...")

