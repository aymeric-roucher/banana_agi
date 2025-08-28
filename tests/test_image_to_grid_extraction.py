import json

from banana_agi.dataset_loader import grid_to_image
from banana_agi.solver import ARCSolver


class TestImageToGridExtraction:
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = ARCSolver()

    def test_basic_image_to_grid_conversion(self):
        """Test basic image to grid conversion with known data."""
        # Simple 3x3 grid
        test_grid = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        # Convert to image and back
        test_image = grid_to_image(test_grid)
        extracted_grid = self.solver.extract_grid_from_generated_image(
            test_image, test_grid
        )

        assert extracted_grid == test_grid, (
            f"Expected {test_grid}, got {extracted_grid}"
        )

    def test_rectangular_grid_conversion(self):
        """Test conversion with rectangular grids."""
        test_grid = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

        test_image = grid_to_image(test_grid)
        extracted_grid = self.solver.extract_grid_from_generated_image(
            test_image, test_grid
        )

        assert extracted_grid == test_grid
        assert len(extracted_grid) == 2
        assert len(extracted_grid[0]) == 5

    def test_large_grid_conversion(self):
        """Test with larger grids."""
        # 10x10 grid with pattern
        test_grid = []
        for i in range(10):
            row = [(i + j) % 10 for j in range(10)]
            test_grid.append(row)

        test_image = grid_to_image(test_grid)
        extracted_grid = self.solver.extract_grid_from_generated_image(
            test_image, test_grid
        )

        assert extracted_grid == test_grid
        assert len(extracted_grid) == 10
        assert len(extracted_grid[0]) == 10

    def test_real_arc_task_data(self):
        """Test with real ARC task data."""
        with open("ARC-AGI-2/data/evaluation/1ae2feb7.json", "r") as f:
            task_data = json.load(f)

        # Test with first training example
        train_input = task_data["train"][0]["input"]
        train_output = task_data["train"][0]["output"]

        # Test input conversion
        input_image = grid_to_image(train_input)
        extracted_input = self.solver.extract_grid_from_generated_image(
            input_image, train_input
        )
        assert extracted_input == train_input

        # Test output conversion
        output_image = grid_to_image(train_output)
        extracted_output = self.solver.extract_grid_from_generated_image(
            output_image, train_output
        )
        assert extracted_output == train_output

        # Test with test example
        test_input = task_data["test"][0]["input"]
        test_output = task_data["test"][0]["output"]

        test_input_image = grid_to_image(test_input)
        extracted_test_input = self.solver.extract_grid_from_generated_image(
            test_input_image, test_input
        )
        assert extracted_test_input == test_input

        test_output_image = grid_to_image(test_output)
        extracted_test_output = self.solver.extract_grid_from_generated_image(
            test_output_image, test_output
        )
        assert extracted_test_output == test_output

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty grid
        empty_grid = []
        result = self.solver.extract_grid_from_generated_image(None, empty_grid)
        assert result is None

        # Single cell grid
        single_cell = [[5]]
        single_image = grid_to_image(single_cell)
        extracted_single = self.solver.extract_grid_from_generated_image(
            single_image, single_cell
        )
        assert extracted_single == single_cell

    def test_color_accuracy(self):
        """Test that all ARC colors are extracted correctly."""
        # Grid with all possible ARC colors (0-9)
        color_grid = [[i] for i in range(10)]

        color_image = grid_to_image(color_grid)
        extracted_colors = self.solver.extract_grid_from_generated_image(
            color_image, color_grid
        )

        assert extracted_colors == color_grid
        for i in range(10):
            assert extracted_colors[i][0] == i, f"Color {i} not extracted correctly"


if __name__ == "__main__":
    # Run tests manually if executed directly
    test_instance = TestImageToGridExtraction()
    test_instance.setup_method()

    try:
        test_instance.test_basic_image_to_grid_conversion()
        print("✅ Basic conversion test passed")

        test_instance.test_rectangular_grid_conversion()
        print("✅ Rectangular grid test passed")

        test_instance.test_large_grid_conversion()
        print("✅ Large grid test passed")

        test_instance.test_real_arc_task_data()
        print("✅ Real ARC data test passed")

        test_instance.test_edge_cases()
        print("✅ Edge cases test passed")

        test_instance.test_color_accuracy()
        print("✅ Color accuracy test passed")

        print("\n🎉 All tests passed! Image-to-grid extraction is working correctly.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
