# 第三方源码快照

快照是从服务器已有工作区选取的必要框架源码副本，用于记录实际使用的实现。整理时保留上游许可证，并在 `source_lock.json` 中记录逐文件 SHA256。

- `groundingdino/`：离线包中的 Open-GroundingDino 训练实现，采用 `models/GroundingDINO`，不是官方的 `groundingdino.models` 包结构，需要单独编译 CUDA 算子。
- `llamafactory/`：离线包中支持 Qwen3-VL 的代码；服务器旧顶层 `LLaMA-Factory/` 的模板集合不同。
- `ms-swift/`：服务器上的 `4.0.0.dev0` 源码。其 GSPO 示例采用序列级重要性采样；原始铁路专用 RL 启动脚本尚未恢复。

逐文件哈希标识所保留的内容，不代表已确认对应某个上游 Git commit。框架依赖与 GPU 运行环境需独立安装，CPU 工作流不导入这些框架。各快照内部的 README、LICENSE 和源码注释保留上游原文，避免改变来源内容与校验值。
