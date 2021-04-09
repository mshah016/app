document.onreadystatechange = function () {
    switch (this.readyState) {
        case "loading":
            break;
        case "interactive":
            var textarea = document.getElementById("user_input");
            var submit = document.getElementById("submit-button");
            submit.onclick = sendToServer;
            break;
        case "complete":
            break;
    }
};

function sendToServer() {
        var textarea = document.getElementById("user_input").value;
        var data = {"content": textarea };
        var str_json = JSON.stringify(data)

        var xhr = new XMLHttpRequest;
        xhr.responseType = "json";

        xhr.onreadystatechange = function () {
            if (this.readyState === 4) {
                if (this.status >= 200 & this.status < 300) {
                    console.log("Success!");
                    console.log("Response: ", xhr.response);
                    document.getElementById("inputted-text").innerHTML = xhr.response['text']
                    document.getElementById("prediction").innerHTML = xhr.response['sentiment']
                } else {
                    console.log("Fail!");
                    console.log("Response: ", xhr.response);
                }
            }
        };

        xhr.open("POST", '/input', true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.send(str_json);
    }

       