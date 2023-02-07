
def write_rttm(SEL_tuples, out_rttm_file):
    
    file_id  = out_rttm_file.split('/')[-1].replace('.rttm', '')
    place_holder = ['SPEAKER', file_id, '1', '0', '0', '<NA>', '<NA>', '0', '<NA>', '<NA>']
   
    with open(out_rttm_file, 'w') as f_output:
        for tup in SEL_tuples:
            place_holder[3] = "%.3f" % tup[0]
            place_holder[4] = "%.3f" % tup[1]
            place_holder[7] = str(tup[2])

            output_string   = ' '.join(place_holder)

            f_output.write(output_string + '\n')

def read_vadfile(vad_file):
    starts, ends = [], []
    with open(vad_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            start, end = line.split(' ')
            starts.append(float(start))
            ends.append(float(end))

    return starts, ends