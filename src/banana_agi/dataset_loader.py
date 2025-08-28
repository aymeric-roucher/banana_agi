import json
import numpy as np
from PIL import Image
import os
import glob

def load_arc_task(task_file):
    """Load a single ARC-AGI task from JSON file."""
    with open(task_file, 'r') as f:
        data = json.load(f)
    return data

def load_all_arc_tasks(dataset_dir):
    """Load all ARC-AGI tasks from a directory containing JSON files."""
    tasks = {}
    json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
    
    for json_file in json_files:
        task_name = os.path.splitext(os.path.basename(json_file))[0]
        tasks[task_name] = load_arc_task(json_file)
    
    return tasks

def grid_to_image(grid, cell_size=20):
    """Convert a grid to an image with colors for each value."""
    colors = [
        (0, 0, 0),        # 0: black
        (0, 116, 217),    # 1: blue
        (255, 65, 54),    # 2: red
        (46, 204, 64),    # 3: green
        (255, 220, 0),    # 4: yellow
        (170, 170, 170),  # 5: gray
        (240, 18, 190),   # 6: magenta
        (255, 133, 27),   # 7: orange
        (127, 219, 255),  # 8: sky blue
        (135, 12, 37),    # 9: maroon
    ]
    
    grid = np.array(grid)
    height, width = grid.shape
    
    img = Image.new('RGB', (width * cell_size, height * cell_size))
    pixels = img.load()
    
    for y in range(height):
        for x in range(width):
            color = colors[grid[y, x]]
            for dy in range(cell_size):
                for dx in range(cell_size):
                    pixels[x * cell_size + dx, y * cell_size + dy] = color
    
    return img

def transform_task_to_images(task_data, output_dir, task_name):
    """Transform input/output couples to images for a single task."""
    task_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    
    # Process training examples
    for i, example in enumerate(task_data.get('train', [])):
        input_img = grid_to_image(example['input'])
        output_img = grid_to_image(example['output'])
        
        input_img.save(os.path.join(task_dir, f'train_{i}_input.png'))
        output_img.save(os.path.join(task_dir, f'train_{i}_output.png'))
    
    # Process test examples
    for i, example in enumerate(task_data.get('test', [])):
        input_img = grid_to_image(example['input'])
        input_img.save(os.path.join(task_dir, f'test_{i}_input.png'))
        
        # Some test examples might not have outputs
        if 'output' in example:
            output_img = grid_to_image(example['output'])
            output_img.save(os.path.join(task_dir, f'test_{i}_output.png'))

def main():
    dataset_dir = 'ARC-AGI-2/data/evaluation'  # Path to the evaluation dataset
    output_dir = 'arc_images'
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        print("Please make sure the ARC-AGI-2 repository is cloned in the current directory.")
        return
    
    print(f"Loading tasks from {dataset_dir}")
    tasks = load_all_arc_tasks(dataset_dir)
    print(f"Found {len(tasks)} tasks")
    
    for task_name, task_data in tasks.items():
        print(f"Processing task: {task_name}")
        transform_task_to_images(task_data, output_dir, task_name)
    
    print(f"Images saved to {output_dir} directory")

if __name__ == "__main__":
    main()