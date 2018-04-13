function greet(name) {
  // if nothing is passed--will set name as undefined
  name = name || '<Your name here>'; // or operator coerces the value that returns true
  console.log('Hello ' + name);
}

greet();
