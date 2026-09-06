# Huo 文档

Huo 是一个用于扇束 CT 正向投影与迭代图像重建的项目，提供 Python / PyTorch 实现和 TypeScript / jax-js 浏览器演示。

- [使用教程（English）](tutorial.md)：几何参数、正向投影、图像重建以及浏览器演示的使用方法。
- [反向投影的数学证明](backward_propagation_proof.md)：残差回填的几何解释、与严格 ART 的等价条件，以及离散迭代的收敛条件和反例。
- [文档维护与发布](publishing.md)：本地预览、添加文章和通过 GitHub Actions 发布文档。

首次使用项目可以从教程开始。若关注 `backward_propagation` 为什么能用于重建，数学文章会从源码对应的算子出发，逐步给出假设与证明。

源码、安装说明和问题反馈见 [GitHub 仓库](https://github.com/zhiqwang/huo)。
