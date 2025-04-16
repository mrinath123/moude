import os

def generate_html(save_dir, output_html_path):
    # Create the base structure for the HTML
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Predictions</title>
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            table, th, td {
                border: 1px solid black;
            }
            th, td {
                padding: 10px;
                text-align: center;
            }
            img {
                width: 100%;
                max-width: 200px;
                height: auto;
            }
        </style>
    </head>
    <body>
        <h1>Segmentation Results</h1>
        <table>
            <tr>
                <th>Original</th>
                <th>Ground Truth</th>
                <th>Predictions</th>
                <th>COTTA Predictions</th>
            </tr>
    """
    
    # Directory paths for original images, ground truth, model predictions, and Cotta predictions
    original_dir = os.path.join(save_dir, 'original_images')
    gt_dir = os.path.join(save_dir, 'gt')
    pred_dir = os.path.join(save_dir, 'model_pred')
    cotta_pred_dir = os.path.join(save_dir, 'cotta_pred')
    
    # Get the list of image filenames
    image_filenames = sorted(os.listdir(original_dir))  # Sort the files to ensure consistent order
    
    # Iterate through the image filenames and append the rows to the HTML table
    for img_name in image_filenames:
        original_img_path = os.path.join('original_images', img_name)  # Relative paths for HTML
        gt_img_path = os.path.join('gt', img_name)
        pred_img_path = os.path.join('model_pred', img_name)
        cotta_pred_img_path = os.path.join('cotta_pred', img_name)
        
        # Add a row for each set of images
        html_content += f"""
        <tr>
            <td><img src="{original_img_path}" alt="Original Image"></td>
            <td><img src="{gt_img_path}" alt="Ground Truth"></td>
            <td><img src="{pred_img_path}" alt="Predictions"></td>
            <td><img src="{cotta_pred_img_path}" alt="COTTA Predictions"></td>
        </tr>
        """

    # Close the table and the HTML document
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Write the HTML content to a file
    with open(output_html_path, 'w') as f:
        f.write(html_content)
    print(f"HTML file generated at {output_html_path}")

# Example usage
save_dir = '/BS/DApt/work/project/segformer_test/segmentation_results'  # Replace with your actual directory
output_html_path = 'output_predictions.html'  # The path where the HTML will be saved
generate_html(save_dir, output_html_path)
