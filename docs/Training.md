## Training using the Ninapro DB4 dataset
Assuming that we want emg-data from FDS, triceps and biceps for the movements `0, 1, 2, 5` in Exercise C. If not then either change the labels or emg_columns lists.

Extract all `S+_E+_A+.mat` files into a single dir with a folder named `csv`, then in GNU Octave or Matlab run
```
function save_emg_labels(emg_tab, label_vec, name)
    labels = [0, 1, 2, 5];
    emg_columns = [9, 10, 11, 12];

    row_mask = any(ismember(label_vec, labels), 3);
    selected_labels = label_vec(row_mask);

    selected_labels(selected_labels == 5) = 4;  % Reduce outputs from model
    emg_data = emg_tab(row_mask, emg_columns);

    % Open file for writing
    fid = fopen(name, 'w');

    % Write header
    fprintf(fid, 'flex1,flex2,bicep,tricep,label\n');

    % Write data row by row
    for i = 1:size(emg_data, 1)
        fprintf(fid, '%f,%f,%f,%f,%d\n', emg_data(i, 1), emg_data(i, 2), emg_data(i, 3), emg_data(i, 4), selected_labels(i));
    endfor

    fclose(fid);

endfunction

files = glob('*.mat')

for i = 1:30
    load(files{i})
    if exercise == 3
        printf("Extracting from subject %d\n", subject)
        save_emg_all(emg, repetition, ['csv/','S', num2str(subject), '_repetition.csv'])
    endif
endfor
```
This will extract the rows in the columns corresponding to the sensors when the label is one we want. Make sure that the csv-files are in a folder inside the `data` directory under project root, such as
```
Workdir
	data
		_name you like_
			S1_*.csv
			...
			S10_*.csv
```