# Firefly Algorithm (FA) Image Segmentation Using Clustering

### This code is free to use. The Matlab and Python codes are available. 

![Firefly Algorithm (FA) Image Segmentation Using Clustering](https://user-images.githubusercontent.com/11339420/176389547-c4a9f480-ee6e-4145-8e90-38d34602b7bb.jpg)

```markdown
# Firefly Algorithm (FA) for Image Segmentation

This repository contains an implementation of the Firefly Algorithm (FA) for image segmentation using clustering. The algorithm is applied to segment an image into clusters based on pixel intensities, effectively grouping similar regions together.

## Features
- **Firefly Algorithm**: A bio-inspired optimization technique used for clustering.
- **Image Processing**: Includes grayscale conversion, histogram equalization, and image reshaping for clustering.
- **Customizable Parameters**: Easy adjustment of the number of clusters and algorithm parameters.
- **Visualization**: Outputs the segmented image and a plot of the algorithm's cost evolution.

## Dependencies
To run the code, you need the following Python libraries:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `scikit-image`

Install the required libraries using pip:
```bash
pip install numpy matplotlib scikit-learn scikit-image
```

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Firefly-Algorithm-Image-Segmentation.git
   cd Firefly-Algorithm-Image-Segmentation
   ```

2. Place your input image (`f.jpg`) in the same directory as the script.

3. Run the Python script:
   ```bash
   python fa_image_segmentation.py
   ```

4. View the results:
   - **Cost Evolution Plot**: Shows the best cost at each iteration.
   - **Segmented Image**: Displays the original image and its segmented version.

## Parameters
You can customize the following parameters in the script:

- `k`: Number of clusters (default is 6).
- `MaxIt`: Maximum number of iterations for the Firefly Algorithm.
- `nPop`: Number of fireflies in the population.
- `gamma`: Light absorption coefficient.
- `beta0`: Attraction coefficient base value.
- `alpha`: Mutation coefficient.

## Example Output
### Input Image
Original image (`f.jpg`):
![Original Image](example_original.jpg)

### Segmented Image
Segmented output:
![Segmented Image](example_segmented.jpg)

### Cost Evolution
The evolution of the best cost during optimization:

![Cost Evolution](example_cost_plot.jpg)

## How It Works
1. **Image Preprocessing**:
   - Converts the input image to grayscale.
   - Adjusts contrast using histogram equalization.
   - Reshapes the image into a vector for clustering.

2. **Firefly Algorithm**:
   - Initializes a population of fireflies with random positions (cluster centers).
   - Iteratively updates positions based on light intensity (cost function).
   - Minimizes the within-cluster distance to find optimal clustering.

3. **Segmentation**:
   - Maps clustered indices back to the original image shape.
   - Creates a segmented image by assigning cluster labels to pixels.

## Contributing
Feel free to fork this repository, make enhancements, and submit a pull request. Contributions are welcome!

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact
For questions or feedback, please contact [your email/username].
```

