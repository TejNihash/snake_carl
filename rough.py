import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

frame_skip = 2

for i in range(1,frame_skip+1):
    if i%(frame_skip) == 0:
        print(i)
    rough = i

print("last",rough)

dist = None

for i in range(3):
    dist_min  = None

    if dist_min<= dist:
        print("this runs")