var person = new Object();
var person = {}; //shorthand to declare 'new Object()'
// or you can declare properties as so:
var Tony = { firstname: 'Tony',
               lastname: 'Alicea',
               address: {
                 street: '111 Main St',
                 city: 'New York',
                 state: 'NY'
               }
             }


function greet(person){
  console.log('Hi ' + person.firstname);
}

greet(Tony);
greet({ firstname: 'Mary', lastname: 'Doe'});

Tony.address2 = {
  street: '333 Second St'
}

// person["firstname"] = "Tony";
// person["lastname"] = "Alicea";
//
// var firstnameProperty = "firstname";
//
// console.log(person); // call object
// console.log(person[firstnameProperty]); // computational member access
// console.log(person.firstname); // 'dot' member access
//
// person.address = new Object();
// person.address.street = "111 Main St";
// person.address.city = "New York";
// person.address.state = "NY";
//
// console.log(person.address.street);
// console.log(person["address"]["street"]);
