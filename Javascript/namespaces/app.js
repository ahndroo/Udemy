var greet = 'Hello!';
var greet = 'Hola';

console.log(greet);

// create container objects for 'greet'
var english = {};
var spanish = {};

// now variables dont override eachother
english.greet = 'Hello';
spanish.greet = 'Hola!';

// english.greetings.greet = 'Hello!'; // won't work, need to init greetings object to give properties
// use either:
// english.greetings = {};
//or
var english = {
  greetings: {
    basic: 'Hello!';
  }
}
