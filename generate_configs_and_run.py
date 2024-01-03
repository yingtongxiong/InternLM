import ast
import os
import subprocess

# 配置参数
model_size_list = [7, 13, 30, 65]
seq_len_list = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
# model_size_list = [7]
# seq_len_list = [4096, 8192]


# 循环处理每个配置
for model_size in model_size_list:
    for seq_len in seq_len_list:
        # 生成目标文件路径
        output_file_path = f"./sim_configs/output_{model_size}_{seq_len}.py"
        config_template_path = f"./configs/{model_size}_template.py"
        config_output_path = f"./configs/{model_size}_{seq_len}_config.py"

        # 检查文件是否存在
        if not os.path.isfile(output_file_path):
            print(f"Output file not found: {output_file_path}")
            continue
        if not os.path.isfile(config_template_path):
            print(f"Config template file not found: {config_template_path}")
            continue

        # 读取output文件中的dict字段内容
        with open(output_file_path, "r") as output_file:
            content = output_file.read()
            try:
                dict_content = ast.literal_eval(content)
            except SyntaxError:
                print(f"Error parsing dict from file: {output_file_path}")
                continue

        # 读取模板文件内容
        with open(config_template_path, "r") as template_file:
            template_content = template_file.read()

        # 替换模板文件中的字段
        config_content = template_content
        for key, value in dict_content.items():
            config_content = config_content.replace(f"{{{key}}}", str(value))

        # 写入配置文件
        with open(config_output_path, "w") as config_file:
            config_file.write(config_content)

        print(f"Configuration file generated: {config_output_path}")

        command = f"srun -p llm_s -N 16 -n 128 --ntasks-per-node=8 --gpus-per-task=1 --time=120 python train.py --config {config_output_path} 2>&1s | tee ./simulator_logs/{model_size}_{seq_len}.log"
        process = subprocess.Popen(command, shell=True, executable="/bin/bash")
        process.wait()

print("Completed!!!")
