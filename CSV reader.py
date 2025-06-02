import csv

def process_csv(input_file, output_file):
    processed_rows = []

    with open(input_file, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        processed_rows.append(headers)

        for row in reader:
            new_row = []
            for cell in row:
                if 'Ip/' in cell:
                    # find start and end positions
                    start = cell.find('Ip/') + len('Ip/')
                    end = cell.find('/', start)
                    if end == -1:
                        end = len(cell)
                    extracted = cell[start:end]
                    cleaned = extracted.replace('-', ' ')
                    new_row.append(cleaned)
                else:
                    new_row.append(cell)
            processed_rows.append(new_row)

   
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(processed_rows)

    print(f"Processed CSV saved as '{output_file}'")

input_csv = 'input.csv'
output_csv = 'output.csv'
process_csv(input_csv, output_csv)