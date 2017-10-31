SkinAI
=============

一个Python Flask WebApp,提供皮肤异常图片上传，取得诊断结果结口

API
---
The app exposes a RESTful API.

* `GET /`: returns a list of available user info.
* `GET /book/<isbn>`: 取得指定用户的诊断结果
* `GET /new`:提供新加用户病例的表单.
* `POST /new/<isbn>` 新加一个指定用户ID病例

Requirements
------------
* mongodb keras tensorflow pillow flask

Usage
-----
$ python server.py


