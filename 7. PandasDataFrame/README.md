
# タイタニック号のデータセット（titanic.csv）

[titanic.csv](https://github.com/jorisvandenbossche/pandas-tutorial/blob/master/data/titanic.csv)


# このハンズオンを動作させるために

現在(2023.07.27)、LangChainにBugがありグラフ表示が行えません。
[langchain.schema.OutputParserException](https://github.com/langchain-ai/langchain/issues/7001)

Pull Requestが発行されていますので、記載の通りLangChainを修正してみます。
(通常こんな事はしませんが、ハンズオンのためやっちゃいます)
[PullReq](https://github.com/langchain-ai/langchain/compare/master...vai0:langchain:master)

PyCharmの左ペインのExternal Librariesから以下のファイルを修正します。
External Libraries/site-packages/langchain/agents/openai_functions_agent/base.py
External Libraries/site-packages/langchain/tools/python/tool.py
