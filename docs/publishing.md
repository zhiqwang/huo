# 文档维护与发布

本站由 `docs/` 下的 Markdown 文件生成，使用 MkDocs Material，发布地址为 [zhiqwang.github.io/huo](https://zhiqwang.github.io/huo/)。行内公式与块公式通过 MathJax 渲染，支持证明文章中的矩阵、分段函数和公式编号。

在仓库根目录执行以下命令，使用 Python 3.12 创建独立的文档环境：

```bash
python3.12 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r requirements-docs.txt
.venv-docs/bin/python -m mkdocs serve
```

本地预览地址为 [http://127.0.0.1:8000/huo/](http://127.0.0.1:8000/huo/)。MkDocs 会在修改 Markdown 时自动重新构建。

发布前使用与工作流相同的严格构建命令：

```bash
.venv-docs/bin/python -m mkdocs build --strict
```

生成的静态文件保存在 `site/`。文档环境和构建产物均被 Git 忽略。

添加文章时，将 Markdown 文件放入 `docs/`，并在根目录 `mkdocs.yml` 的 `nav` 中添加导航项。站内链接使用相对 Markdown 路径，例如 `[教程](tutorial.md)`；指向 `docs/` 外部源码的链接使用 GitHub 文件地址，避免发布后指向不存在的站内路径。

数学公式可以使用 `$...$` 和独占行的 `$$...$$`。公式脚本使用固定版本的 MathJax；配置位于 `docs/javascripts/mathjax.js`，渲染脚本通过 jsDelivr 加载。

首次配置仓库时，在 [Settings → Pages](https://github.com/zhiqwang/huo/settings/pages) 中把 **Build and deployment → Source** 设为 **GitHub Actions**。使用自定义 Pages 工作流之前需要启用这一发布来源，见 [GitHub 官方说明](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

[Deploy documentation 工作流](https://github.com/zhiqwang/huo/actions/workflows/docs.yml) 的行为如下：

- Pull request 修改文档、MkDocs 配置、文档依赖或该工作流时，执行严格构建检查。
- 同类变更推送到 `main` 时，构建站点、上传 Pages artifact，再部署到 `github-pages` 环境。
- 在 Actions 页面选择 **Run workflow**，并选择 `main`，可以手动重新发布；选择其他分支只执行构建。

部署任务只在构建通过后运行，使用 GitHub 自带的 `GITHUB_TOKEN` 和 OIDC 权限，无须配置个人访问令牌。部署状态与站点链接会显示在工作流的 `github-pages` 环境中。
