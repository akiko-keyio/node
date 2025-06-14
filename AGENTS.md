> 本指南为 Codex 生成 Python 软件包时的“行为约束文档”，强调低耦合、高内聚、遵循 SOLID 且避免过度设计，并在整个生命周期内贯彻 YAGNI 与极简主义。([digitalocean.com][1], [geeksforgeeks.org][2])

## 摘要

* **Ruff 一站式静态分析+格式化**：以 `ruff check` + `ruff format` 统一 lint 与代码风格，速度远超 Black/isort/flake8 组合。([docs.astral.sh][3], [github.com][4])
* **uv / pip** 负责环境与依赖；两者均可 **直接驱动 `pyproject.toml`** 工作流，且 uv 与 pip 完全兼容。([docs.astral.sh][5], [github.com][6])
* 设计层面坚持 **SOLID**，同时用 **YAGNI** 限制前期膨胀，保证实现仅含最小必要元素。([digitalocean.com][1], [reddit.com][7])
* 测试首选 **pytest + coverage**；CI 中设置 `--fail-under` 阈值确保质量门槛。([docs.pytest.org][8], [stackoverflow.com][9], [coverage.readthedocs.io][10])

---

## 1. 设计原则

| 原则      | 在包中的落地方式                                                      |
| ------- | ------------------------------------------------------------- |
| **S**RP | 每个模块/函数只承担单一责任，变更原因唯一。([digitalocean.com][1])                 |
| **O**CP | 通过抽象基类或 `typing.Protocol` 开放扩展、封闭修改。                          |
| **L**SP | 子类必须可无痛替换父类；优先组合而非继承。                                         |
| **I**SP | 将大接口拆分为细粒度协议，避免“胖接口”。                                         |
| **D**IP | 高层依赖抽象而非具体，实现依赖注入或 `functools.partial`。([arjancodes.com][11]) |

> **关键提醒：** 若某原则实施后导致模板代码急剧增加且用户价值不变，即视为过度设计，应回退。([geeksforgeeks.org][2])

---

## 2. 编码规范

1. **Ruff 为唯一格式与 Lint 工具**

   * `ruff format .` 保证代码自动对齐与换行。
   * `ruff check .` 开启默认规则；必要时在 `pyproject.toml` 内微调。([docs.astral.sh][3], [docs.astral.sh][12])
2. **类型标注**

   * 所有公共 API 必须完整注解，CI 触发 `mypy --strict`。([packaging.python.org][13])
3. **函数体 ≤ 20 行**，最多一级嵌套；复杂逻辑拆分为私有辅助函数。
4. **异常与日志**

   * 核心层直接抛出异常；边界层捕获并记录。
5. **Docstring**

   * 模块级说明“是什么 & 为什么”；函数级说明“做什么 & 参数/返回”。

---

## 3. 依赖管理

| 工具      | 适用场景         | 关键命令                                                                       |
| ------- | ------------ | -------------------------------------------------------------------------- |
| **uv**  | 高速、可复现的构建与安装 | `uv pip install -r requirements.txt`([docs.astral.sh][5], [astral.sh][14]) |
| **pip** | 经典、稳定、广泛支持   | `python -m pip install .`([reddit.com][15], [packaging.python.org][13])    |

* **最小依赖集合**：能用标准库解决的场景禁止引入第三方库（YAGNI）。([geeksforgeeks.org][2])
* 所有依赖声明于 `project.dependencies`（运行时）与 `project.optional-dependencies.dev`（开发时）段落内。([packaging.python.org][13])

---

## 4. 测试与质量保障

1. **pytest**：覆盖全部公共功能路径；保持测试代码同样简洁。([docs.pytest.org][8], [emimartin.me][16])
3. **property-based 测试**：对纯函数可选用 Hypothesis 提升稳健性。

---

## 5. 自动化与 CI

* **pre-commit**：配置 `ruff`, `mypy`, `pytest` 钩子，在本地即阻止劣质提交。
* **GitHub Actions**：

  ```yaml
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
    - run: uv pip install -e .[dev]  # 或 python -m pip ...
    - run: ruff format --check .
    - run: ruff check .
    - run: mypy .
    - run: pytest --cov=src
    - run: coverage report --fail-under=90
  ```

([docs.astral.sh][5], [stackoverflow.com][9], [emimartin.me][16])

---

## 6. 生成策略

> 在编写任何代码前先自问：**“如果现在删掉这段实现，用户功能是否受损？”**——若答案为否，立即移除。

* **先写测试，再写实现**（TDD-light）。
* **小步提交**：每次 commit 仅覆盖单一逻辑变更，信息格式 `feat|fix|docs: <scope>`。
* **显式导出**：所有对外符号必须写入 `__all__`，避免 API 泄漏。
* **文档即代码的一部分**：更新 Public API 时同步更新 README 与 docstring。

---


[1]: https://www.digitalocean.com/community/conceptual-articles/s-o-l-i-d-the-first-five-principles-of-object-oriented-design?utm_source=chatgpt.com "SOLID Design Principles Explained: Building Better Software ..."
[2]: https://www.geeksforgeeks.org/what-is-yagni-principle-you-arent-gonna-need-it/?utm_source=chatgpt.com "What is YAGNI principle (You Aren't Gonna Need It)? - GeeksforGeeks"
[3]: https://docs.astral.sh/ruff/formatter/?utm_source=chatgpt.com "The Ruff Formatter - Astral Docs"
[4]: https://github.com/astral-sh/ruff?utm_source=chatgpt.com "astral-sh/ruff: An extremely fast Python linter and code formatter ... - GitHub"
[5]: https://docs.astral.sh/uv/pip/compatibility/?utm_source=chatgpt.com "Compatibility with pip | uv - Astral Docs"
[6]: https://github.com/astral-sh/uv?utm_source=chatgpt.com "astral-sh/uv: An extremely fast Python package and project ... - GitHub"
[7]: https://www.reddit.com/r/ExperiencedDevs/comments/11vonwg/yagni_is_a_good_principle_but_many_devs_miss_the/?utm_source=chatgpt.com "\"YAGNI\" is a good principle, but many devs miss the point and conflate it ..."
[8]: https://docs.pytest.org/en/stable/explanation/goodpractices.html?utm_source=chatgpt.com "Good Integration Practices - pytest documentation"
[9]: https://stackoverflow.com/questions/59420123/is-there-a-standard-way-to-fail-pytest-if-test-coverage-falls-under-x?utm_source=chatgpt.com "Is there a standard way to fail pytest if test coverage falls under x%"
[10]: https://coverage.readthedocs.io/?utm_source=chatgpt.com "Coverage.py — Coverage.py 7.9.1 documentation"
[11]: https://arjancodes.com/blog/dependency-inversion-principle-in-python-programming/?utm_source=chatgpt.com "Mastering Dependency Inversion in Python Coding | ArjanCodes"
[12]: https://docs.astral.sh/ruff/?utm_source=chatgpt.com "Ruff - Astral Docs"
[13]: https://packaging.python.org/tutorials/managing-dependencies/?utm_source=chatgpt.com "Managing Application Dependencies - Python Packaging User Guide"
[14]: https://astral.sh/blog/uv?utm_source=chatgpt.com "uv: Python packaging in Rust - Astral"
[15]: https://www.reddit.com/r/Python/comments/1gphzn2/a_completeish_guide_to_dependency_management_in/?utm_source=chatgpt.com "A complete-ish guide to dependency management in Python - Reddit"
[16]: https://emimartin.me/pytest_best_practices?utm_source=chatgpt.com "Pytest best practices - Emiliano Martin"
