//fetch('http://127.0.0.1:5000/input', {
//    mode: 'no-cors',
//}).then(function (response) {
//    return response.json()
//}).then(function (json) {
//    console.log('GET response JSON: ');
//    console.log(json)
//})


console.log("this is a test string")

fetch('/hello')
    .then(function (response) {
        return response.text();
    }).then(function (text) {
        console.log('GET response text:');
        console.log(text); // Print the greeting as text
    });

// Send the same request
fetch('/hello')
    .then(function (response) {
        return response.json(); // But parse it as JSON this time
    })
    .then(function (json) {
        console.log('GET response as JSON:');
        console.log(json); // Here’s our JSON object
    })