# 文档维护与发布

本站由 `docs/` 下的 Markdown 文件生成，使用 Sphinx 和 PyTorch Sphinx Theme 2，发布地址为 [zhiqwang.github.io/huo](https://zhiqwang.github.io/huo/)。行内公式与块公式通过 MathJax 渲染，支持证明文章中的矩阵、分段函数和公式编号。

在仓库根目录执行以下命令，使用 uv 和 Python 3.12 安装文档依赖：

```bash
uv sync --locked --only-group docs --python 3.12
uv run --locked --only-group docs sphinx-autobuild docs site
```

本地预览地址为 [http://127.0.0.1:8000/](http://127.0.0.1:8000/)。sphinx-autobuild 会在修改文档时自动重新构建。

发布前使用与工作流相同的严格构建命令：

```bash
uv run --locked --only-group docs sphinx-build -b html -W --keep-going docs site
```

生成的静态文件保存在 `site/`。uv 环境 `.venv/` 和构建产物均被 Git 忽略。文档依赖在 `pyproject.toml` 的 `docs` 依赖组中定义，并通过 `uv.lock` 锁定；修改依赖后运行 `uv lock`。

添加文章时，将 Markdown 文件放入 `docs/`，并在 `docs/index.md` 的 `toctree` 中添加导航项。站内链接使用相对 Markdown 路径，例如 `[教程](tutorial.md)`；指向 `docs/` 外部源码的链接使用 GitHub 文件地址，避免发布后指向不存在的站内路径。

数学公式可以使用 `$...$` 和独占行的 `$$...$$`。Sphinx 使用 MathJax 渲染公式；相关配置位于 `docs/conf.py`。

首次配置仓库时，在 [Settings → Pages](https://github.com/zhiqwang/huo/settings/pages) 中把 **Build and deployment → Source** 设为 **GitHub Actions**。使用自定义 Pages 工作流之前需要启用这一发布来源，见 [GitHub 官方说明](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

[Deploy documentation 工作流](https://github.com/zhiqwang/huo/actions/workflows/docs.yml) 的行为如下：

- Pull request 修改文档、Sphinx 配置、文档依赖或该工作流时，执行严格构建检查。
- 同类变更推送到 `main` 时，构建站点、上传 Pages artifact，再部署到 `github-pages` 环境。
- 在 Actions 页面选择 **Run workflow**，并选择 `main`，可以手动重新发布；选择其他分支只执行构建。

部署任务只在构建通过后运行，使用 GitHub 自带的 `GITHUB_TOKEN` 和 OIDC 权限，无须配置个人访问令牌。部署状态与站点链接会显示在工作流的 `github-pages` 环境中。
