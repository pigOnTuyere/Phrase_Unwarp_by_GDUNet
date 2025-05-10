import inspect
import os
import json

def save_log(args, main_func, model):
    """Save model metadata and training parameters as a .txt file"""

    # 确保输出目录存在
    dir_model = os.path.join(args.root, args.output)
    if os.path.exists(dir_model):  # 检查路径是否存在
        raise FileExistsError(f"路径 '{dir_model}' 已经存在！")  # 如果路径存在，抛出异常
    else:
        os.makedirs(dir_model)

    width = 80  # 每行宽度限制

    # 获取 main_func 和 model 的代码
    main_func_code = inspect.getsource(main_func)
    model_code = inspect.getsource(model)

    # 构造日志内容
    log_content = f"""
    Model Name: UNet
    Epochs: {args.epochs}
    Batch Size: {args.batch_size}
    Initial Learning Rate: {args.lr}
    Root Directory: {args.root}
    Output Directory: {dir_model}
    Input Folder: {args.input}
    Ground Truth Folder: {args.gt}
    Main Function Code:{main_func_code}
    Model Info:{model_code}
                    """

    # 保存日志到 txt 文件
    log_path = os.path.join(dir_model, "log.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(log_content)

    log = {
        "model_name": "UNet",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "initail learning rate": args.lr,
        "root": args.root,
        "output_directory": dir_model,
        "input_folder": args.input,
        "ground_truth_folder": args.gt,
        "main func": main_func_code,
        "model_info": model_code,
    }
    with open(os.path.join(dir_model, "log.json"), 'w') as f:
        json.dump(log, f, indent=4)