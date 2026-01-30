import imageio.v2 as imageio

VIDEO_PATH = "video/sport-union-1190-20260112-fussball-00.11.10.680-00.12.56.790.mkv"

reader = imageio.get_reader(VIDEO_PATH)

meta = reader.get_meta_data()
print("FPS:", meta.get("fps"))
print("Size:", meta.get("size"))

# read one frame
frame = reader.get_data(0)
print("Frame shape:", frame.shape)

reader.close()
