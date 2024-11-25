import numpy as np
from scipy.fftpack import dct, idct
import cv2
from collections import Counter
import heapq
import json
import os
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Button, Scale, HORIZONTAL, Canvas, Radiobutton, IntVar
from PIL import Image, ImageTk

image_path = None
Quality_factor = 50
is_gray = True  # Default to grayscale
canvas_images = []  # Store references to canvas objects
canvas_image_refs = []  # Keep references to PhotoImage objects to avoid garbage collection
current_x_offset = 100  # Offset for image placement

#      _ ____  _____ ____ 
#     | |  _ \| ____/ ___|
#  _  | | |_) |  _|| |  _ 
# | |_| |  __/| |__| |_| |
#  \___/|_|   |_____\____|

def dct_2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def quantize(block, Q, A):
    M = (50/Q) * A
    BLOCK =  (np.round(block / M)).astype(int)
    return BLOCK

def divide_into_blocks(image):
    h, w = image.shape
    h_blocks = h // 8
    w_blocks = w // 8
    blocks = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = image[i*8:(i+1)*8, j*8:(j+1)*8]
            blocks.append(block)
    return blocks

def zigzag_scan(block):
    zigzag_order = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]
    return [block[i, j] for i, j in zigzag_order]

def collect_ac_coefficients(quantized_blocks):
    coefficients = []
    for block in quantized_blocks:
        zigzag_coefficients = zigzag_scan(block)
        coefficients.extend(zigzag_coefficients[1:])
    return coefficients

def collect_dc_coefficients(quantized_blocks):
    return [int(block[0, 0]) for block in quantized_blocks]

def huffman_encoding(data):
    data_non_zero = [x for x in data if x != 0]
    frequency = Counter(data_non_zero)
    heap = [[int(weight), [int(symbol), ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huffman_dict = {int(k): str(v) for k, v in dict(heapq.heappop(heap)[1:]).items()}
    return huffman_dict

def encode_ac_coefficients(ac_coefficients, ac_huffman_table):
    encoded_ac = []
    run_length = 0
    for coefficient in ac_coefficients:
        coefficient = int(coefficient)
        if coefficient == 0:
            run_length += 1
        else:
            huffman_code = ac_huffman_table[coefficient]
            size = len(huffman_code)
            encoded_ac.append((int(run_length), size, str(huffman_code)))
            run_length = 0
    if run_length > 0:
        encoded_ac.append((0, 0, '0'))  # EOB symbol
    return encoded_ac

def huffman_encoding_dc1(data):
    frequency = Counter(data)
    heap = [[weight, [int(symbol), ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huffman_dict = {int(k): str(v) for k, v in dict(heapq.heappop(heap)[1:]).items()}
    return huffman_dict

def huffman_encoding_dc(dc_coefficients):
    dc_coeff_encoded_data = [int(dc_coefficients[0])]
    for i in range(1, len(dc_coefficients)):
        diff = int(dc_coefficients[i]) - int(dc_coefficients[i-1])
        dc_coeff_encoded_data.append(diff)

    dc_huffman_table = huffman_encoding_dc1(dc_coeff_encoded_data[1:])
    return dc_huffman_table

def jpeg_encoder(image, Q, A):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float32) - 128
    blocks = divide_into_blocks(image)
    dct_blocks = [dct_2d(block) for block in blocks]
    quantized_blocks = [quantize(block, Q, A) for block in dct_blocks]
    dc_coefficients = collect_dc_coefficients(quantized_blocks)
    ac_coefficients = collect_ac_coefficients(quantized_blocks)
    dc_huffman_table = huffman_encoding_dc(dc_coefficients)
    ac_huffman_table = huffman_encoding(ac_coefficients)
    encoded_blocks = []
    prev_dc = 0

    for i, block in enumerate(quantized_blocks):
        zigzag_coefficients = zigzag_scan(block)
        if i == 0:
            dc_encoded = int(zigzag_coefficients[0])
        else:
            diff = zigzag_coefficients[0] - prev_dc
            dc_encoded = str(dc_huffman_table[diff])

        prev_dc = zigzag_coefficients[0]
        ac_coefficients = zigzag_coefficients[1:]
        ac_encoded = encode_ac_coefficients(ac_coefficients, ac_huffman_table)
        encoded_blocks.append([dc_encoded, ac_encoded])
    return encoded_blocks, dc_huffman_table, ac_huffman_table

def save_jpeg_encoded_file(encoded_blocks, dc_huffman_table, ac_huffman_table, Quality_factor, image_shape, color_mode, output_file):
    jpeg_data = {
        "image_shape": image_shape,
        "color_mode": color_mode,
        "quality_factor": Quality_factor,
        "dc_huffman_table": dc_huffman_table,
        "ac_huffman_table": ac_huffman_table,
        "encoded_blocks": encoded_blocks
    }
    
    with open(output_file, "wb") as f:
        pickle.dump(jpeg_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def read_jpeg_encoded_file(input_file):
    with open(input_file, "rb") as f:
        jpeg_data = pickle.load(f)
    return (jpeg_data["image_shape"], jpeg_data["quality_factor"],jpeg_data["dc_huffman_table"], jpeg_data["ac_huffman_table"],jpeg_data["encoded_blocks"])

def decode_dc_coefficients(encoded_dc, dc_huffman_table):
    reverse_dc_huffman_table = {v: k for k, v in dc_huffman_table.items()}
    dc_differences = [encoded_dc[0]]
    for code in encoded_dc[1:]:
        diff = int(reverse_dc_huffman_table[code])
        dc_differences.append(diff)

    dc_coefficients = [dc_differences[0]]
    for i in range(1, len(dc_differences)):
        dc_coefficients.append(dc_coefficients[i-1] + dc_differences[i])

    return dc_coefficients

def decode_ac_coefficients(encoded_ac, ac_huffman_table):
    reverse_ac_huffman_table = {v: k for k, v in ac_huffman_table.items()}
    ac_coefficients = []

    for run_length, size, huffman_code in encoded_ac:
        if run_length == 0 and size == 0:
            break
        coefficient = reverse_ac_huffman_table[huffman_code]
        ac_coefficients.extend([0] * run_length)
        ac_coefficients.append(coefficient)

    while len(ac_coefficients) < 63:
        ac_coefficients.append(0)

    return ac_coefficients

def decode_blocks(encoded_blocks, dc_huffman_table, ac_huffman_table):
    decoded_blocks = []
    dc_encoded = [encoded_blocks[0][0]]
    for i in range(1,len(encoded_blocks)):
        dc_encoded.append(encoded_blocks[i][0])
    dc_coefficients = decode_dc_coefficients(dc_encoded, dc_huffman_table)

    for i in range(len(dc_coefficients)):
        dc_coefficient = dc_coefficients[i]
        ac_encoded = encoded_blocks[i][1]
        ac_coefficients = decode_ac_coefficients(ac_encoded, ac_huffman_table)
        zigzag_coefficients = [dc_coefficient] + ac_coefficients
        block = np.zeros((8, 8), dtype=int)
        zigzag_order = [
            (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
            (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
            (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
            (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
            (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
            (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
            (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
            (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
        ]
        for (i, j), coefficient in zip(zigzag_order, zigzag_coefficients):
            block[i, j] = coefficient

        decoded_blocks.append(block)
    return decoded_blocks

def dequantize(block, Q, A):
    M = (50 / Q) * A
    return block * M

def idct_2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def reconstruct_patches(decoded_blocks, Q, A):
    image_patches = []
    for block in decoded_blocks:
        dequantized_block = dequantize(block, Q, A)
        spatial_block = idct_2d(dequantized_block)
        restored_block = np.round(spatial_block + 128).astype(np.uint8)
        image_patches.append(restored_block)
    return image_patches

def combine_patches(image_patches, image_shape):
    h, w = image_shape
    h_blocks = h // 8
    w_blocks = w // 8

    reconstructed_image = np.zeros((h_blocks*8, w_blocks*8), dtype=np.uint8)
    for i in range(h_blocks):
        for j in range(w_blocks):
            patch_idx = i * w_blocks + j
            reconstructed_image[i*8:(i+1)*8, j*8:(j+1)*8] = image_patches[patch_idx]    
    return reconstructed_image

def save_image_and_check_size(image, output_file, input_file):
    cv2.imwrite(output_file, image)
    file_size_bytes = os.path.getsize(output_file)
    file_size_kb = file_size_bytes / 1024.0
    input_file_size_bytes = os.path.getsize(input_file)
    input_file_size_kb = input_file_size_bytes / 1024.0

# Quantization matrix for Q = 50
A = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]]).astype(int)

def jpeg_compressor(input_image, Quality_factor, color=False):
    global A    
    if __name__ == "__main__":
        original_shape = input_image.shape[:2]
        height, width = original_shape
        new_height = (height // 8) * 8
        new_width = (width // 8) * 8
        input_image = input_image[:new_height, :new_width]
        encoded_blocks, dc_huffman_table, ac_huffman_table = jpeg_encoder(input_image, Quality_factor, A)

    save_jpeg_encoded_file(
        encoded_blocks=encoded_blocks,
        dc_huffman_table=dc_huffman_table,
        ac_huffman_table=ac_huffman_table,
        Quality_factor=Quality_factor,  # Quantization matrix
        image_shape=input_image.shape,
        color_mode="grayscale",  # Change to "color" for color images
        output_file="compressed_image.bin"
    )

    input_file = "compressed_image.bin"
    image_shape_2, quality_factor_2, dc_huffman_table_2, ac_huffman_table_2, encoded_blocks_2 = read_jpeg_encoded_file(input_file)
    decoded_blocks = decode_blocks(encoded_blocks_2, dc_huffman_table_2, ac_huffman_table_2)
    image_patches = reconstruct_patches(decoded_blocks, quality_factor_2, A)
    final_image = combine_patches(image_patches, image_shape_2[:2])
    if color != True:
        output_path = "Compressed_image.jpg"
        cv2.imwrite('Compressed_image.jpg', final_image)
    else:
        return final_image

def color_jpeg_compressor(image, Quality_factor):
    original_shape = image.shape[:2]
    height, width = original_shape
    new_height = (height // 8) * 8
    new_width = (width // 8) * 8
    image = image[:new_height, :new_width]
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    y = ycbcr_image[:, :, 0]
    cb = ycbcr_image[:, :, 1]
    cr = ycbcr_image[:, :, 2]

    cb_downsampled = cv2.resize(cb, (cb.shape[1] // 2, cb.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cr_downsampled = cv2.resize(cr, (cr.shape[1] // 2, cr.shape[1] // 2), interpolation=cv2.INTER_AREA)

    processed_y = jpeg_compressor(y, Quality_factor, color=True)
    processed_cb = jpeg_compressor(cb_downsampled, Quality_factor, color=True)
    processed_cr = jpeg_compressor(cr_downsampled, Quality_factor, color=True)

    cb_upsampled = cv2.resize(processed_cb, (cb.shape[1], cb.shape[0]), interpolation=cv2.INTER_AREA)
    cr_upsampled = cv2.resize(processed_cr, (cr.shape[1], cr.shape[0]), interpolation=cv2.INTER_AREA)
    ycbcr_compressed = cv2.merge((processed_y, cb_upsampled, cr_upsampled))

    compressed_image = cv2.cvtColor(ycbcr_compressed, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite('Compressed_image.jpg', compressed_image)

#  _____ _    _       _               ____ _   _ ___ 
# |_   _| | _(_)_ __ | |_ ___ _ __   / ___| | | |_ _|
#   | | | |/ / | '_ \| __/ _ \ '__| | |  _| | | || | 
#   | | |   <| | | | | ||  __/ |    | |_| | |_| || | 
#   |_| |_|\_\_|_| |_|\__\___|_|     \____|\___/|___|

def clear_images():
    global canvas_images, canvas_image_refs, current_x_offset
    for item in canvas_images:
        canvas.delete(item)
    canvas_images = []
    canvas_image_refs = []
    current_x_offset = 100
    
def load_image():
    global image_path
    clear_images()
    filetypes = [("Image files", "*.jpg;*.jpeg;*.png;*.pgm;*.tiff")]  # Add more extensions as per need
    image_path = filedialog.askopenfilename(filetypes=filetypes)
    if not image_path:
        messagebox.showerror("Error", "No file selected")
        return

    display_image(image_path, "Original Image")

def set_quality_factor(val):
    global Quality_factor
    Quality_factor = int(val)

def set_image_type(value):
    global is_gray
    is_gray = bool(int(value))

def compress_image():
    global image_path, output_file, is_gray, Quality_factor
    if not image_path:
        messagebox.showerror("Error", "Please select an image first!")
        return
    input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if is_gray:
        jpeg_compressor(input_image, Quality_factor, color=False)
    else:
        color_jpeg_compressor(input_image, Quality_factor)
    messagebox.showinfo("Success", f"Image compressed and saved to Compressed_image.jpg")
    display_image('./Compressed_image.jpg', "Compressed Image")
    
def display_image(image_path, title):
    global canvas_images, canvas_image_refs, current_x_offset
    image = Image.open(image_path)
    image.thumbnail((400, 400))
    image = ImageTk.PhotoImage(image)
    canvas_image_refs.append(image)
    canvas_image = canvas.create_image(current_x_offset + 200, 250, image=image, anchor="center")
    canvas_images.append(canvas_image)  # Store reference to the canvas object
    title_label = canvas.create_text(current_x_offset + 200, 50, text=title, font=("Arial", 16), fill="blue")
    canvas_images.append(title_label)
    current_x_offset += 450  # Add spacing for the next image

root = tk.Tk()
root.title("JPEG Encoder/Decoder")
root.geometry("900x500")
left_panel = tk.Frame(root, width=180, bg="lightgrey")
left_panel.pack(side="left", fill="y")
load_button = Button(left_panel, text="Load Image", command=load_image)
load_button.pack(pady=10, padx=10)
quality_slider = Scale(left_panel, from_=10, to=100, orient=HORIZONTAL, label="Quality Factor", command=set_quality_factor)
quality_slider.set(50)
quality_slider.pack(pady=10, padx=10)

image_type_label = tk.Label(left_panel, text="Type of Image", bg="lightgrey", font=("Arial", 12))
image_type_label.pack(pady=5)

image_type_var = IntVar(value=1)
grayscale_button = Radiobutton(left_panel, text="Grayscale", variable=image_type_var, value=1, command=lambda: set_image_type(1))
grayscale_button.pack(pady=5)

color_button = Radiobutton(left_panel, text="Color", variable=image_type_var, value=0, command=lambda: set_image_type(0))
color_button.pack(pady=5)

compress_button = Button(left_panel, text="Compress Image", command=compress_image)
compress_button.pack(pady=10, padx=10)

clear_button = Button(left_panel, text="Clear Images", command=clear_images)
clear_button.pack(pady=10, padx=10)

exit_button = Button(left_panel, text="Exit", command=root.quit)
exit_button.pack(pady=10, padx=10)

canvas = Canvas(root, bg="white", width=720, height=500)
canvas.pack(side="right", fill="both", expand=True)

root.mainloop()