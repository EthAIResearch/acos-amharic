import os

def filter_tsv(input_path, output_path):
    input_count = 0
    output_count = 0
    total_input_quads = 0
    total_output_quads = 0
    
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8", newline="") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            input_count += 1
            parts = line.split("\t")
            text = parts[0]
            quads = parts[1:]
            
            total_input_quads += len(quads)
            
            # Filter: Keep only quads where both aspect and opinion spans are NOT "-1,-1"
            explicit_quads = []
            for quad in quads:
                if not quad.strip():
                    continue
                quad_parts = quad.strip().split(" ")
                if len(quad_parts) != 4:
                    continue
                a_span, category, sentiment, o_span = quad_parts
                if a_span != "-1,-1" and o_span != "-1,-1":
                    explicit_quads.append(quad)
            
            # If we have at least one explicit quad, keep the sentence
            if explicit_quads:
                output_count += 1
                total_output_quads += len(explicit_quads)
                fout.write(text + "\t" + "\t".join(explicit_quads) + "\n")
                
    print(f"Processed {os.path.basename(input_path)}:")
    print(f"  Input sentences: {input_count} | Output sentences: {output_count}")
    print(f"  Input quads:     {total_input_quads} | Output quads:     {total_output_quads}")

def main():
    raw_dir = r"c:\Users\azari\Documents\Dev_projects\amharic-acos\data\raw"
    out_dir = r"c:\Users\azari\Documents\Dev_projects\amharic-acos\data\raw_explicit"
    
    os.makedirs(out_dir, exist_ok=True)
    
    splits = ["train", "dev", "test"]
    for split in splits:
        in_file = os.path.join(raw_dir, f"amharic_quad_{split}.tsv")
        out_file = os.path.join(out_dir, f"amharic_quad_{split}.tsv")
        if os.path.exists(in_file):
            filter_tsv(in_file, out_file)
        else:
            print(f"Warning: {in_file} does not exist.")

if __name__ == "__main__":
    main()
