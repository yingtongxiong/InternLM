import copy
import os
import subprocess

name = "./configs/"
root_names = ["7B_train_", "13B_train_", "30B_train_"]
model_size = ["7B", "13B", "30B"]
micro_bsz = [1, 2, 4, 8, 16, 32, 64]
sp = ["none", "megatron", "flash-attn", "intern", "intern"]
intern_overlap = [False, False, False, True, False]
checkpoint = [False, True]

for idx, root_name in enumerate(root_names):
    # 指定要创建的文件夹路径
    folder_path = name + root_name[:-1]

    # 使用os.mkdir()创建文件夹
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    file_name = name + f"{model_size[idx]}_template.py"

    with open(file_name, "r") as f:
        lines = f.readlines()
        origin_line = "".join(lines)
        for mb in micro_bsz:
            for i, sp_mode in enumerate(sp):
                for ckpt in checkpoint:
                    line = copy.copy(origin_line)
                    line = line.replace("{micro_bsz}", str(mb))
                    line = line.replace("{sp}", f'"{sp_mode}"')
                    line = line.replace("{intern_overlap}", str(intern_overlap[i]))
                    line = line.replace("{checkpoint}", str(ckpt))
                    output_file_name = (
                        str(mb)
                        + "_"
                        + str(sp_mode)
                        + "_overlap_"
                        + str(intern_overlap[i])
                        + "_ckpt_"
                        + str(ckpt)
                        + ".py"
                    )
                    write_file = folder_path + "/" + output_file_name
                    with open(write_file, "w") as file:
                        file.write(line)

                    log_name = root_name + "_" + output_file_name[:-3]

                    print(log_name)
                    command = f"srun -p llm_t -x HOST-10-140-66-5 -N 8 -n 64 --ntasks-per-node=8 --gpus-per-task=1 --time=20 python train.py --config {write_file} --profiling 2>&1 | tee ./fstp_logs/{log_name}.log"
                    process = subprocess.Popen(command, shell=True, executable="/bin/bash")
                    process.wait()
