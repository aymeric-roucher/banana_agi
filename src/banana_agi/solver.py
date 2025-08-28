import os
from io import BytesIO

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from google import genai
from PIL import Image
from tqdm import tqdm

from banana_agi.dataset_loader import grid_to_image, load_all_arc_tasks

load_dotenv()

FONT_SIZE = 16


class ARCSolver:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        self.client = genai.Client(api_key=api_key)

    def image_to_grid(self, image_path, expected_rows=None, expected_cols=None):
        """Convert an image back to a grid for comparison."""
        if isinstance(image_path, str):
            img = Image.open(image_path).convert("RGB")
        else:
            # If it's already a PIL Image
            img = image_path.convert("RGB")

        # ARC color mapping (RGB values to grid values)
        color_to_value = {
            (0, 0, 0): 0,  # black
            (0, 116, 217): 1,  # blue
            (255, 65, 54): 2,  # red
            (46, 204, 64): 3,  # green
            (255, 220, 0): 4,  # yellow
            (170, 170, 170): 5,  # gray
            (240, 18, 190): 6,  # magenta
            (255, 133, 27): 7,  # orange
            (127, 219, 255): 8,  # sky blue
            (135, 12, 37): 9,  # maroon
        }

        width, height = img.size

        # If expected dimensions are provided, calculate cell size accordingly
        if expected_rows and expected_cols:
            cell_width = width / expected_cols
            cell_height = height / expected_rows
        else:
            # Default behavior - assume 20px cells
            cell_width = cell_height = 20
            expected_cols = width // int(cell_width)
            expected_rows = height // int(cell_height)

        grid = []
        for y in range(expected_rows):
            row = []
            for x in range(expected_cols):
                # Sample pixel from center of cell
                pixel_x = int(x * cell_width + cell_width / 2)
                pixel_y = int(y * cell_height + cell_height / 2)

                # Ensure coordinates are within bounds
                pixel_x = min(pixel_x, width - 1)
                pixel_y = min(pixel_y, height - 1)

                pixel_color = img.getpixel((pixel_x, pixel_y))

                # Find closest color match
                closest_value = 0
                min_distance = float("inf")
                for color, value in color_to_value.items():
                    distance = sum((a - b) ** 2 for a, b in zip(pixel_color, color))
                    if distance < min_distance:
                        min_distance = distance
                        closest_value = value

                row.append(closest_value)
            grid.append(row)

        return grid

    def extract_grid_from_generated_image(self, image, expected_grid):
        """Extract grid from API-generated image based on expected output dimensions."""
        expected_rows = len(expected_grid)
        expected_cols = len(expected_grid[0]) if expected_grid else 0

        if expected_rows == 0 or expected_cols == 0:
            return None

        return self.image_to_grid(image, expected_rows, expected_cols)

    def parse_grid_from_response(self, response_text):
        """Parse grid data from Gemini response."""
        lines = response_text.strip().split("\n")
        grid = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse as comma-separated values
            if "," in line:
                row = [int(x.strip()) for x in line.split(",") if x.strip().isdigit()]
                if row and all(0 <= x <= 9 for x in row):
                    grid.append(row)

            # Parse as space-separated values
            elif " " in line:
                row = [int(x) for x in line.split() if x.isdigit()]
                if row and all(0 <= x <= 9 for x in row):
                    grid.append(row)

            # Parse as individual digits
            elif line.replace(" ", "").isdigit():
                row = [int(x) for x in line.replace(" ", "") if x.isdigit()]
                if row and all(0 <= x <= 9 for x in row):
                    grid.append(row)

        return grid if grid else None

    def create_recap_image(
        self,
        task_name: str,
        train_examples: list[dict],
        test_example: dict,
        prediction: list[list[int]],
        test_idx: int,
    ):
        """Create a recap image showing training examples and a single test case using matplotlib."""
        # Create figure with gridspec layout
        fig = plt.figure(figsize=(16, 10))

        # Create gridspec: left side for training examples, right side for test case
        gs = gridspec.GridSpec(
            3,
            4,
            figure=fig,
            width_ratios=[1, 1, 1, 1],
            height_ratios=[1, 1, 1],
            hspace=0.15,
            wspace=0.15,
        )

        # Add main title
        fig.suptitle(
            f"Task: {task_name} - Test Case {test_idx + 1}",
            fontsize=FONT_SIZE + 2,
            fontweight="bold",
        )

        # Left side: Training examples (2x3 grid)
        for i, train_ex in enumerate(train_examples[:3]):  # Limit to 3 examples
            # Training input (left column)
            ax_input = fig.add_subplot(gs[i, 0])
            train_input_img = grid_to_image(train_ex["input"])
            ax_input.imshow(np.array(train_input_img))
            ax_input.set_title(
                f"Train {i + 1} Input",
                fontsize=FONT_SIZE,
                color="blue",
                fontweight="bold",
            )
            ax_input.axis("off")

            # Training output (second column)
            ax_output = fig.add_subplot(gs[i, 1])
            train_output_img = grid_to_image(train_ex["output"])
            ax_output.imshow(np.array(train_output_img))
            ax_output.set_title(
                f"Train {i + 1} Output",
                fontsize=FONT_SIZE,
                color="green",
                fontweight="bold",
            )
            ax_output.axis("off")

        # Right side: Test case (third and fourth columns)
        # Test input
        ax_test_input = fig.add_subplot(gs[0, 2:])
        test_input_img = grid_to_image(test_example["input"])
        ax_test_input.imshow(np.array(test_input_img))
        ax_test_input.set_title(
            "Test Input", fontsize=FONT_SIZE, color="blue", fontweight="bold"
        )
        ax_test_input.axis("off")

        # Predicted output
        ax_pred_output = fig.add_subplot(gs[1, 2:])
        if prediction:
            predicted_output_img = grid_to_image(prediction)
            ax_pred_output.imshow(np.array(predicted_output_img))
            ax_pred_output.set_title(
                "Predicted Output",
                fontsize=FONT_SIZE,
                color="orange",
                fontweight="bold",
            )
        else:
            ax_pred_output.text(
                0.5,
                0.5,
                "No prediction generated",
                ha="center",
                va="center",
                fontsize=FONT_SIZE,
                color="red",
                fontweight="bold",
            )
            ax_pred_output.set_title(
                "Predicted Output",
                fontsize=FONT_SIZE,
                color="orange",
                fontweight="bold",
            )
        ax_pred_output.axis("off")

        # Expected output
        ax_expected = fig.add_subplot(gs[2, 2:])
        expected_output_img = grid_to_image(test_example["output"])
        ax_expected.imshow(np.array(expected_output_img))
        ax_expected.set_title(
            "Expected Output", fontsize=FONT_SIZE, color="green", fontweight="bold"
        )
        ax_expected.axis("off")

        # Save the figure
        os.makedirs("predictions/recap_images", exist_ok=True)
        filename = f"predictions/recap_images/{task_name}_{test_idx}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)  # Close figure to free memory
        print(f"Saved recap image: {filename}")

    def solve_task(self, task_data: dict, task_name: str):
        """Solve an ARC task using Gemini vision model."""
        train_examples = task_data.get("train", [])
        test_examples = task_data.get("test", [])

        if not train_examples or not test_examples:
            return []

        predictions = []

        for test_idx, test_example in enumerate(test_examples):
            # Create images for this specific test case
            test_input_img = grid_to_image(test_example["input"])

            # Create training example images
            train_images = []
            for i, train_ex in enumerate(train_examples):
                train_input_img = grid_to_image(train_ex["input"])
                train_output_img = grid_to_image(train_ex["output"])
                train_images.extend([train_input_img, train_output_img])

            # Prepare prompt
            prompt = f"""You are solving an ARC (Abstraction and Reasoning Corpus) task. 

I will show you training examples as pairs of input-output grids (as images), followed by a test input grid. Your task is to identify the pattern from the training examples and apply it to generate the correct output image for the test input.

Training examples ({len(train_examples)} pairs):
"""

            for i in range(len(train_examples)):
                prompt += f"Example {i + 1}: Input -> Output\n"

            prompt += """
Test input (generate the output for this):

Rules:
1. Study each training example in depth to understand the transformation pattern. It can be symetries, translations, rotations, any of these can be affected by the shapes in presence as if they were physical objects.
2. Write a 5-line summary of what the transformation pattern is : the basis of it, caveats, any other observations.
3. Apply the same pattern to the test input to generate the output. Respond ONLY with the image of the output, with correct colors. To generate it, just start from the test input image and modify according to the transformation pattern.

Output grid:"""

            # Create content with images
            content_parts = [prompt]

            # Add training images
            for img in train_images:
                content_parts.append(img)

            # Add test input
            content_parts.append(test_input_img)

            response = self.client.models.generate_content(
                model="gemini-2.5-flash-image-preview", contents=content_parts
            )
            if hasattr(response, "text"):
                print(response.text)

            predicted_grid = None
            output_image = None

            # Extract image from response
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    output_image = Image.open(BytesIO(part.inline_data.data))
                    os.makedirs("predictions", exist_ok=True)
                    output_image.save(
                        f"predictions/single_outputs/{task_name}_{test_idx}.png"
                    )
                    break

            # Extract grid from generated image
            expected_output = test_example["output"]
            predicted_grid = self.extract_grid_from_generated_image(
                output_image, expected_output
            )

            predictions.append(predicted_grid)

        # Create recap images for each test case
        for test_idx, (test_example, prediction) in enumerate(
            zip(test_examples, predictions)
        ):
            self.create_recap_image(
                task_name, train_examples, test_example, prediction, test_idx
            )

        return predictions

    def calculate_accuracy(self, predictions, ground_truth):
        """Calculate accuracy of predictions."""
        if not predictions or not ground_truth:
            return 0.0

        total_tests = len(ground_truth)
        correct_predictions = 0

        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            if pred is None:
                continue

            if (
                len(pred) == len(truth)
                and len(pred) > 0
                and len(pred[0]) == len(truth[0])
            ):
                if pred == truth:
                    correct_predictions += 1
                    print(f"Test {i + 1}: CORRECT")
                else:
                    print(f"Test {i + 1}: INCORRECT")
            else:
                print(f"Test {i + 1}: INCORRECT (dimension mismatch)")

        accuracy = correct_predictions / total_tests
        return accuracy


def main():
    solver = ARCSolver()

    # Load ARC dataset
    dataset_dir = "ARC-AGI-2/data/evaluation"
    tasks = load_all_arc_tasks(dataset_dir)

    print(f"Loaded {len(tasks)} tasks")

    total_accuracy = 0
    total_tasks = 0

    # Test on first 5 tasks
    task_items = list(tasks.items())[:10]

    for task_name, task_data in tqdm(task_items):
        print(f"\n{'=' * 50}")
        print(f"Solving task: {task_name}")
        print(f"{'=' * 50}")

        predictions = solver.solve_task(task_data, task_name)
        ground_truth = [test_ex["output"] for test_ex in task_data.get("test", [])]

        accuracy = solver.calculate_accuracy(predictions, ground_truth)
        print(f"\nTask {task_name} accuracy: {accuracy:.2%}")

        total_accuracy += accuracy
        total_tasks += 1

    overall_accuracy = total_accuracy / total_tasks if total_tasks > 0 else 0
    print(f"\n{'=' * 50}")
    print(f"Overall accuracy: {overall_accuracy:.2%} ({total_tasks} tasks)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
