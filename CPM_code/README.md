To run cpm, simply do:
```bash
python run_cpm.py -json settings.json -job jobs &> logs.txt &
```
An example of CPM input is provided in CPM_input_example/.

To generate .json and .mat needed by the CPM code, you can use generate_json_and_mat.ipynb.

To analyze CPM output, you can use analysis_and_plotting.ipynb and get_sig_edges.ipynb.