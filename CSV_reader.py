import csv

def extract_ip_terms(input_file, output_file):
 
    extracted_terms = []

    try:
        with open(input_file, mode='r', newline='', encoding='utf-8') as f_in:
            reader = csv.reader(f_in)
            headers = next(reader)  # skip header row (we don't need to output it)
            
            for row in reader:
                for cell in row:
                    if 'ip/' in cell:
                        start = cell.find('ip/') + len('ip/')
                        end = cell.find('/', start)
                        if end == -1:
                            end = len(cell)
                        raw_term = cell[start:end]
                        cleaned = raw_term.replace('-', ' ')
                        extracted_terms.append(cleaned)
                       
    except FileNotFoundError:
        print(f"Error: Cannot open '{input_file}'. Please check that it exists.")
        return

    
    with open(output_file, mode='w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        for term in extracted_terms:
            writer.writerow([term])

    print(f"Extraction complete. {len(extracted_terms)} terms written to '{output_file}'.")


# ─── Main ───
input_csv  = 'RD_RD_TBL_BI_SCIENCEATA_2025_10KUSER_SAMP_3_wmt.csv'
output_csv = 'output.csv'
extract_ip_terms(input_csv, output_csv)