// JSON - JavaScript Object Notation
// inspired by object literal syntax

var objectLiteral = {
  firstname: 'Mary',
  isAProgrammer: true
}

console.log(objectLiteral);

// in JSON, the format is very similar to JS object literal Notation

// {
//   "firstname": "Mary",
//   "isAProgrammer": true
// }

// to convert to JSON string
console.log(JSON.stringify(objectLiteral));

//take string in proper JSON and convert to JS object
var jsonValue = JSON.parse('{ "firstname": "Mary", "isAProgrammer":true}');
console.log(jsonValue);
