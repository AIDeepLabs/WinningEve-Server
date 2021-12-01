
## IP Addresses

When users do not specify secretKey then we can just use there IP address as a secret key for a limited amount of usages.



```python
from flask import request
from flask import jsonify

@app.route("/get_my_ip", methods=["GET"])
def get_my_ip():
    return jsonify({'ip': request.remote_addr}), 200
```

As for nginx, it sends the real IP address under HTTP_X_FORWARDED_FOR so make sure you don't end up with localhost for each request.