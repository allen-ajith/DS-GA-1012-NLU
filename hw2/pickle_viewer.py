import pickle
import os

def list_pickle_files(directory):
    # List all files with .pkl, .pickle, or .p extensions in the given directory
    return [f for f in os.listdir(directory) if f.endswith(('.pkl', '.pickle', '.p'))]

def print_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print(f"\nContents of {file_path}:")
        print(data)
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")

def main():
    # Ask the user for a directory path
    directory = input("Enter the path to the directory containing pickle files: ").strip()
    
    # Check if the directory exists
    if not os.path.isdir(directory):
        print("The provided path is not a valid directory. Please try again.")
        return

    # List pickle files in the specified directory
    pickle_files = list_pickle_files(directory)
    
    if not pickle_files:
        print("No pickle files found in the specified directory.")
        return

    print("\nPickle files in the specified directory:")
    for i, file in enumerate(pickle_files, 1):
        print(f"{i}. {file}")

    while True:
        choice = input("\nEnter the number of the file to print (or 'q' to quit): ")
        if choice.lower() == 'q':
            break
        try:
            index = int(choice) - 1
            if 0 <= index < len(pickle_files):
                # Construct full file path and print its contents
                file_path = os.path.join(directory, pickle_files[index])
                print_pickle_file(file_path)
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")

if __name__ == "__main__":
    main()
