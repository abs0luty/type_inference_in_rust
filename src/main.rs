use std::{
    collections::HashMap,
    iter::{zip, Peekable},
    sync::Arc,
};

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum Expression {
    Lambda {
        parameter: String,
        value: Box<Self>,
    },
    Apply {
        callee: Box<Self>,
        argument: Box<Self>,
    },
    Variable {
        name: String,
    },
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum Token {
    Lambda,
    OpenParent,
    CloseParent,
    Dot,
    Name(String),
}

pub fn tokenize(source: &str) -> Vec<Token> {
    let mut chars = source.chars().peekable();
    let mut tokens = Vec::new();

    while let Some(c) = chars.next() {
        match c {
            '(' => tokens.push(Token::OpenParent),
            ')' => tokens.push(Token::CloseParent),
            '\\' => tokens.push(Token::Lambda),
            '.' => tokens.push(Token::Dot),
            _ => {
                if c.is_whitespace() {
                    // ignore
                } else if c.is_alphabetic() || c == '_' {
                    let mut name = String::new();

                    name.push(c);

                    while let Some(&c) = chars.peek() {
                        if c.is_alphanumeric() || c == '_' {
                            name.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }

                    tokens.push(Token::Name(name));
                } else {
                    panic!("unexpected character `{}`", c);
                }
            }
        }
    }

    tokens
}

struct ParseState<I>
where
    I: Iterator<Item = Token>,
{
    tokens: Peekable<I>,
}

impl<I> ParseState<I>
where
    I: Iterator<Item = Token>,
{
    fn new(tokens: I) -> Self {
        Self {
            tokens: tokens.peekable(),
        }
    }

    fn parse_parenthesized_expression(&mut self) -> Expression {
        let inner_expression = self.parse_expression();

        if !matches!(self.tokens.next(), Some(Token::CloseParent)) {
            panic!("expected `)`");
        }

        inner_expression
    }

    fn parse_primary_expression(&mut self) -> Expression {
        match self.tokens.next() {
            Some(Token::Lambda) => self.parse_lambda(),
            Some(Token::OpenParent) => self.parse_parenthesized_expression(),
            Some(Token::Name(name)) => Expression::Variable { name },
            _ => {
                panic!("expected expression")
            }
        }
    }

    fn parse_expression(&mut self) -> Expression {
        let mut left = self.parse_primary_expression();

        while !matches!(self.tokens.peek(), Some(Token::CloseParent) | None) {
            let argument = self.parse_primary_expression();

            left = Expression::Apply {
                callee: Box::new(left),
                argument: Box::new(argument),
            };
        }

        left
    }

    fn parse_lambda(&mut self) -> Expression {
        let parameter = match self.tokens.next() {
            Some(Token::Name(name)) => name,
            _ => panic!("expected identifier"),
        };

        if !matches!(self.tokens.next(), Some(Token::Dot)) {
            panic!("expected `.`");
        }

        let value = self.parse_primary_expression();

        Expression::Lambda {
            parameter,
            value: Box::new(value),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Type {
    Constructor(TypeConstructor),
    Variable(TypeVariable),
}

impl Type {
    fn substitute(&self, substitutions: &HashMap<TypeVariable, Arc<Type>>) -> Arc<Type> {
        match self {
            Type::Constructor(TypeConstructor { name, generics }) => {
                Arc::new(Type::Constructor(TypeConstructor {
                    name: name.clone(),
                    generics: generics
                        .iter()
                        .map(|t| t.substitute(substitutions))
                        .collect(),
                }))
            }
            Type::Variable(TypeVariable(i)) => {
                if let Some(t) = substitutions.get(&TypeVariable(*i)) {
                    t.substitute(substitutions)
                } else {
                    Arc::new(self.clone())
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]

struct TypeConstructor {
    name: String,
    generics: Vec<Arc<Type>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct TypeVariable(usize);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Constraint {
    Eq { left: Arc<Type>, right: Arc<Type> },
}

impl TypeVariable {
    fn occurs_in(&self, ty: Arc<Type>, substitutions: &HashMap<TypeVariable, Arc<Type>>) -> bool {
        match ty.as_ref() {
            Type::Variable(v @ TypeVariable(i)) => {
                if let Some(substitution) = substitutions.get(&v) {
                    if substitution.as_ref() != &Type::Variable(*v) {
                        return self.occurs_in(substitution.clone(), substitutions);
                    }
                }

                self.0 == *i
            }
            Type::Constructor(TypeConstructor { generics, .. }) => {
                for generic in generics {
                    if self.occurs_in(generic.clone(), substitutions) {
                        return true;
                    }
                }

                false
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct Environment {
    variables: HashMap<String, Arc<Type>>,
}

impl Environment {
    fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, PartialEq, Eq)]
struct InferenceContext<'env> {
    constraints: Vec<Constraint>,
    environment: &'env mut Environment,
    last_type_variable_index: usize,
}

impl<'env> InferenceContext<'env> {
    fn new(environment: &'env mut Environment) -> Self {
        Self {
            constraints: Vec::new(),
            environment,
            last_type_variable_index: 0,
        }
    }

    fn fresh_type_variable(&mut self) -> TypeVariable {
        self.last_type_variable_index += 1;
        TypeVariable(self.last_type_variable_index)
    }

    fn type_placeholder(&mut self) -> Arc<Type> {
        Arc::new(Type::Variable(self.fresh_type_variable()))
    }

    fn infer_type(&mut self, expression: &Expression) -> Arc<Type> {
        match expression {
            Expression::Variable { name } => self
                .environment
                .variables
                .get(name)
                .unwrap_or_else(|| panic!("unbound variable: {}", name))
                .clone(),
            Expression::Lambda { parameter, value } => {
                let parameter_type = self.type_placeholder();

                self.environment
                    .variables
                    .insert(parameter.clone(), parameter_type.clone());

                let value_type = self.infer_type(value);

                Arc::new(Type::Constructor(TypeConstructor {
                    name: "Function".to_owned(),
                    generics: vec![parameter_type, value_type],
                }))
            }
            Expression::Apply { callee, argument } => {
                let callee_type = self.infer_type(callee);
                let argument_type = self.infer_type(argument);

                let return_type = self.type_placeholder();

                self.constraints.push(Constraint::Eq {
                    left: callee_type.clone(),
                    right: Arc::new(Type::Constructor(TypeConstructor {
                        name: "Function".to_owned(),
                        generics: vec![argument_type.clone(), return_type.clone()],
                    })),
                });

                return_type
            }
        }
    }

    fn solve(self) -> HashMap<TypeVariable, Arc<Type>> {
        let mut substitutions = HashMap::new();

        for constraint in self.constraints {
            match constraint {
                Constraint::Eq { left, right } => {
                    unify(left, right, &mut substitutions);
                }
            }
        }

        substitutions
    }
}

fn unify(left: Arc<Type>, right: Arc<Type>, substitutions: &mut HashMap<TypeVariable, Arc<Type>>) {
    match (left.as_ref(), right.as_ref()) {
        (
            Type::Constructor(TypeConstructor {
                name: name1,
                generics: generics1,
            }),
            Type::Constructor(TypeConstructor {
                name: name2,
                generics: generics2,
            }),
        ) => {
            assert_eq!(name1, name2);
            assert_eq!(generics1.len(), generics2.len());

            for (left, right) in zip(generics1, generics2) {
                unify(left.clone(), right.clone(), substitutions);
            }
        }
        (Type::Variable(TypeVariable(i)), Type::Variable(TypeVariable(j))) if i == j => {}
        (_, Type::Variable(v @ TypeVariable(..))) => {
            if let Some(substitution) = substitutions.get(&v) {
                unify(left, substitution.clone(), substitutions);
                return;
            }

            assert!(!v.occurs_in(left.clone(), substitutions));
            substitutions.insert(*v, left);
        }
        (Type::Variable(v @ TypeVariable(..)), _) => {
            if let Some(substitution) = substitutions.get(&v) {
                unify(right, substitution.clone(), substitutions);
                return;
            }

            assert!(!v.occurs_in(right.clone(), substitutions));
            substitutions.insert(*v, right);
        }
    }
}

macro_rules! tconst {
    ($name:literal,$($generic:expr),*) => {
        Arc::new(Type::Constructor(TypeConstructor {
            name: $name.to_owned(),
            generics: vec![$($generic),*],
        }))
    };
    ($name:literal) => { tconst!($name,) };
}

macro_rules! tvar {
    ($i:literal) => {
        Arc::new(Type::Variable(TypeVariable($i)))
    };
}

fn main() {
    let mut environment = Environment::new();

    environment.variables.insert(
        "add".to_owned(),
        tconst!(
            "Function",
            tconst!("int"),
            tconst!("Function", tconst!("int"), tconst!("int"))
        ),
    );

    let expression = ParseState::new(tokenize("\\a.(add a a)").into_iter()).parse_expression();

    let mut context = InferenceContext::new(&mut environment);
    let ty = context.infer_type(&expression);
    let substitutions = context.solve();

    println!("{:?}", ty.substitute(&substitutions));
}
