# lodge-api

도복 분리 알고리즘 API

## Quick Start

### 환경 구축

```shell
conda env create -f test_env.yaml
conda activate lodge
```

### 코드 실행

```shell
uvicorn main:app --reload --host=0.0.0.0 --port=8000
```

### API 관련: Postman Collection

* /inference

입력 - 드론 촬영 이미지

출력 - 분리 마스크 이미지(Base64)

```json
{
	"info": {
		"_postman_id": "ed0cc36f-1471-4eb9-8995-83589b333557",
		"name": "DNA + Drone Lodge Demo Collection",
		"schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
		"_exporter_id": "21790939"
	},
	"item": [
		{
			"name": "Main",
			"request": {
				"method": "GET",
				"header": []
			},
			"response": []
		},
		{
			"name": "Inference",
			"request": {
				"method": "POST",
				"header": [],
				"body": {
					"mode": "formdata",
					"formdata": [
						{
							"key": "files",
							"type": "file",
							"src": ""
						}
					]
				},
				"url": {
					"raw": "http://localhost:8000/inference",
					"protocol": "http",
					"host": [
						"localhost"
					],
					"port": "8000",
					"path": [
						"inference"
					]
				}
			},
			"response": []
		}
	]
}
```