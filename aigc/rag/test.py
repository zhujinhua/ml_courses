"""
Author: jhzhu
Date: 2024/8/24
Description: 
"""


# Import statements

# Functions and Classes

def main():
    input_file_path = '/Users/jhzhu/Downloads/USCD26/validate_data.json'
    output_file_path = '/Users/jhzhu/Downloads/output/validate_data.json'
    import json

    formatted_data = []

    try:
        # Open the input file in read mode
        with open(input_file_path, 'r', encoding='utf-8') as f:
            # Load the large JSON file as a list of conversations
            conversations = json.load(f)

            # Iterate over each conversation in the list
            for conversation in conversations:
                # Extract patient's question and doctor's response
                if len(conversation) < 2:
                    print(f"Skipping conversation with insufficient data: {conversation}")
                    continue

                    # Initialize a history list
                history = []

                # If the conversation length is greater than 2, populate history
                if len(conversation) > 2:
                    for i in range(0, len(conversation) - 2, 2):
                        human_instruction = conversation[i][3:]
                        model_response = conversation[i + 1][3:]
                        history.append([human_instruction, model_response])

                # The last two elements are the current instruction and output
                instruction = conversation[-2] [3:] # Second last element as the instruction
                output = conversation[-1][3:]  # Last element as the output

                # Create a formatted entry
                if history:
                    formatted_entry = {
                        "instruction": instruction,
                        "output": output,
                        "history": history
                    }
                else:
                    formatted_entry = {
                        "instruction": instruction,
                        "output": output
                    }

                # Add the formatted entry to the list
                formatted_data.append(formatted_entry)

        # Step 3: Write the formatted data to a new JSON file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)

        print(f"Data has been successfully written to {output_file_path}")

    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
    except FileNotFoundError:
        print(f"Input file {input_file_path} not found.")
    except MemoryError:
        print("MemoryError: The file is too large to fit into memory. Consider processing the file in smaller chunks.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
