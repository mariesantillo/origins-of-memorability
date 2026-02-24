import os
from PIL import Image
import torch
import pandas as pd
from resmem import ResMem, transformer


model = ResMem(pretrained=True)
model.eval()

image_folder = "/home/mariesantillo/resmem/bathsong/"

results = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg")):
        image_path = os.path.join(image_folder, filename)


        img = Image.open(image_path).convert("RGB")
        image_x = transformer(img)

        with torch.no_grad():
            input_tensor = image_x.view(-1, 3, 227, 227)
            prediction = model(input_tensor)

        # Convert to numpy array
        pred_value = prediction.item() if prediction.numel() == 1 else prediction.squeeze().tolist()

        # Append results, save filename by hust the fraeme number
        filename = filename.split("_")[1]
        filename = filename.split(".")[0]
        #arrange by frame number
        results.append({
            "filename": filename,
            "prediction": pred_value
        })
# Sort results by frame number
results.sort(key=lambda x: int(x["filename"].split(".")[0]))
# Save results to CSV

df = pd.DataFrame(results)
df.to_csv("bathsong_prediction.csv", index=False)

print("Predictions saved to bathsong_prediction.csv")
