import os
import argparse

def rename_images_in_directory(directory_path):

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    
    all_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                all_files.append(os.path.join(root, file))
    
    for idx, file_path in enumerate(all_files):
        directory = os.path.dirname(file_path)
        
        new_filename = os.path.join(directory, f"{idx}.jpg")
        
        os.rename(file_path, new_filename)
        print(f'Renamed: {file_path} -> {new_filename}')

def main():
    parser = argparse.ArgumentParser(description='Rename images in a directory to 0.jpg, 1.jpg, 2.jpg, etc.')
    parser.add_argument('directory', type=str, help='The directory where the images are located')
    
    args = parser.parse_args()
    
    # Call the renaming function
    rename_images_in_directory(args.directory)

if __name__ == '__main__':
    main()
