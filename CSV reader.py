import csv

def process_csv(RD_RD_TBL_BI_SCIENCEATA_2025_10KUSER_SAMP_3_wmt.csv, output_file):
    processed_rows = []

    with open(RD_RD_TBL_BI_SCIENCEATA_2025_10KUSER_SAMP_3_wmt.csv, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        processed_rows.append(headers)

        for row in reader:
            new_row = []
            for cell in row:
                if 'ip/' in cell:
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

RD_RD_TBL_BI_SCIENCEATA_2025_10KUSER_SAMP_3_wmt.csv = 'input.csv'
output_csv = 'output.csv'
process_csv(RD_RD_TBL_BI_SCIENCEATA_2025_10KUSER_SAMP_3_wmt.csv, output_csv)