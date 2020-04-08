
# Work in progress. This documentation being improved

## Index

- [Main idea](main-idea)
- [Installation](installation)
- [Structure of the repo](structure-of-the-repo)
- [Start using](start-using)
- [Useful links](#useful-links)

## Main idea

This is a software that handles the next scenarios

- Download and preprocess historical data for stock markets
- Create Machine learning model for prediction future price movement
- Tune hyperparameters of the model and data preprocessing 
- Use UI to manage training/tuning jobs
- Save results of every execution for future analysis
- Pick best runs and deploy for productive trading

All together this solution covers end-to-end process.

## Installation

So far there is no pip package published so the only way now is to clone this repo

In order to launch optimization jobs you would need to 

### Prepare data

- download `orderlog` files for needed tickers from [here](ftp-от-zerich)
- convert it from `qsh` to `bin` format via [this tool](format-conversion)

### Store data

Either local filesystem (fast, not scalable) or S3 storage

### Place to store experiments and results

Create a Mongo instance and to add credentials to `./py/config.py`. [Some hosting options with free tiers](db)

## Structure of the repo

- `./py` folder has python scripts with main logic
- `./flask_ui` scripts that do start a server with UI for scheduling experiments and tracking their results
- `./docker` has Dockerfiles for lua tests (bot) and for running tuning jobs

## Start using

The easiest way is to add experiments via UI on locally hosted Flask server and locally launch a Docker container that will start worker. And worker will pick the next job that is ready for processing

## Useful links

### Data

- [List of MOEX tickers](https://www.moex.com/ru/derivatives/)
- [Sample tickerRTS-3.19](https://www.moex.com/ru/contract.aspx?code=RTS-3.19)
- [Finam export](https://www.finam.ru/profile/mosbirzha-fyuchersy/rts-3-18-rih8/export/?market=17&em=454183&code=RIH8&apply=0&df=18&mf=0&yf=2019&from=18.01.2019&dt=18&mt=0&yt=2019&to=18.01.2019&p=7&f=RIH8_190118_190118&e=.txt&cn=RIH8&dtf=1&tmf=1&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1)
- [Hydra](https://stocksharp.ru/products/hydra/)

### Useful libraries

- [TA-lib](https://github.com/mrjbq7/ta-lib)
- [TA list of indicators](https://mrjbq7.github.io/ta-lib/funcs.html)
- [Stockstats python module](https://pythondata.com/stockstats-python-module-various-stock-market-statistics-indicators/)
- [Backtrader](https://www.backtrader.com)
- [Backtesting Systematic Trading Strategies in Python: Considerations and Open Source Frameworks](https://www.quantstart.com/articles/backtesting-systematic-trading-strategies-in-python-considerations-and-open-source-frameworks)
- [Habr finam](https://habr.com/en/post/332700/)

### Parameters tuning

- [Benchmarking](http://ash-aldujaili.github.io/blog/2018/04/01/coco-bayesopt/)
- [Reddit post](https://www.reddit.com/r/MachineLearning/comments/4g2rnu/bayesian_optimization_for_python/)
- https://scikit-optimize.github.io
- [Scikit optimize examples](https://github.com/scikit-optimize/scikit-optimize/blob/master/examples/bayesian-optimization.ipynb)
- [BOHB: ROBUST AND EFFICIENT HYPERPARAMETER OPTIMIZATION AT SCALE](https://www.automl.org/blog_bohb/)
- [HpBandSter](https://github.com/automl/HpBandSter)
 
### Trade history

- [От «ЦЕРИХ» (Plaza II)](http://zerich.qscalp.ru/)
- [ftp от Zerich](ftp://ftp.zerich.com/pub/Terminals/QScalp/History/)
- ftp://athistory.zerich.com/
- [От «ФИНАМ» (Plaza II)](http://finam.qscalp.ru/)
- [От «Scalping.Pro»](http://erinrv.qscalp.ru/)
- [Архив данных](http://qsh.qscalp.ru/)
- [Format conversion](https://github.com/StockSharp/Qsh2Bin)

### Bot creation

- [Trades from Quik via ODBC](https://kbrobot.ru/mysql.html/)
- [Py quik](https://github.com/dv-lebedev/py-quik)
- [Pandas from MySQL](https://pythondata.com/quick-tip-sqlalchemy-for-mysql-and-pandas/)
- [oAuth service](https://auth0.com/docs/quickstart/backend/python/02-using)
- [tweak mysql odbc](https://forum.quik.ru/forum11/topic3264/)
- [Открытие обучение](https://www.opentrainer.ru/videos/lenta-sdelok-ee-interpretatsiya-i-torgovye-signaly/)
- [TP and SL explained](https://www.opentrainer.ru/articles/teyk-profit-i-stop-limit-v-quik-7/)
- [Lua Quik RPC Python](https://github.com/Enfernuz/quik-lua-rpc)
#### LUA
- [Good site on luaq](http://luaq.ru/getFuturesHolding.html)
- [Quik Lua](https://quikluacsharp.ru)
- [Lua rocks](https://stackoverflow.com/questions/33006269/compiling-luasocket-cannot-open-file-string-h/38176102#38176102)

### Engineering and infrastructure

#### DB

- [Mongo cms Keystone](http://demo.keystonejs.com)
- [cloudclustershosting](https://clients.cloudclusters.io/database/mongodb/c47df899a7964e65ab8a41e01f552758/overview)
- [Mongo hosting](https://www.mongoclusters.com)
- [Mongo dump/restore](https://docs.mongodb.com/manual/)
- http://mms.litixsoft.de/index.php?lang=en#
- [Authentification](https://developer.okta.com)
- [CRUD UI](https://www.tutorialrepublic.com/snippets/preview.php?topic=bootstrap&file=crud-data-table-for-database-with-modal-form)
- [Tabulator UI](http://tabulator.info)

- [SQLite browser](https://inloop.github.io/sqlite-viewer/)

#### Flask admin

 - [formatters samples](https://blog.sneawo.com/blog/2017/02/10/flask-admin-formatters-examples/)

#### VPS

- https://www.ionos.com/servers/vps
- https://my.interserver.net/view_vps?id=229942&link=vnc&type=browser_vnc
- https://clients.stronghosting.net/clientarea.php?action=productdetails&id=8013

#### Deployment
 - [AWS chalice](https://chalice.readthedocs.io/en/latest/)

