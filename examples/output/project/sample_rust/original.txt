//! Sample Rust module for reproduction testing.
//!
//! Tests structs, traits, generics, and error handling.

use std::collections::HashMap;
use std::fmt;

// Structs
#[derive(Debug, Clone)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
    pub is_active: bool,
}

#[derive(Debug, Clone)]
pub struct Product {
    pub sku: String,
    pub name: String,
    pub price: f64,
    pub quantity: u32,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Order {
    pub id: String,
    pub user_id: u64,
    pub items: Vec<OrderItem>,
    pub total: f64,
    pub status: OrderStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrderStatus {
    Pending,
    Processing,
    Shipped,
    Delivered,
}

#[derive(Debug, Clone)]
pub struct OrderItem {
    pub product_id: String,
    pub quantity: u32,
    pub unit_price: f64,
}

// Error type
#[derive(Debug)]
pub enum AppError {
    NotFound(String),
    ValidationError(String),
    DatabaseError(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::NotFound(msg) => write!(f, "Not found: {}", msg),
            AppError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            AppError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
        }
    }
}

impl std::error::Error for AppError {}

// Result type alias
pub type Result<T> = std::result::Result<T, AppError>;

// Trait for entities
pub trait Entity {
    type Id;
    fn id(&self) -> Self::Id;
}

impl Entity for User {
    type Id = u64;
    fn id(&self) -> Self::Id {
        self.id
    }
}

impl Entity for Order {
    type Id = String;
    fn id(&self) -> Self::Id {
        self.id.clone()
    }
}

// Generic repository
pub struct Repository<T: Entity + Clone> {
    items: HashMap<String, T>,
}

impl<T: Entity + Clone> Repository<T>
where
    T::Id: ToString,
{
    pub fn new() -> Self {
        Repository {
            items: HashMap::new(),
        }
    }

    pub fn add(&mut self, item: T) {
        let key = item.id().to_string();
        self.items.insert(key, item);
    }

    pub fn get(&self, id: &str) -> Option<&T> {
        self.items.get(id)
    }

    pub fn get_all(&self) -> Vec<&T> {
        self.items.values().collect()
    }

    pub fn delete(&mut self, id: &str) -> bool {
        self.items.remove(id).is_some()
    }

    pub fn count(&self) -> usize {
        self.items.len()
    }
}

// Functions
pub fn create_user(id: u64, name: &str, email: &str) -> User {
    User {
        id,
        name: name.to_string(),
        email: email.to_string(),
        is_active: true,
    }
}

pub fn calculate_order_total(items: &[OrderItem]) -> f64 {
    items
        .iter()
        .map(|item| item.quantity as f64 * item.unit_price)
        .sum()
}

pub fn validate_email(email: &str) -> Result<()> {
    if !email.contains('@') {
        return Err(AppError::ValidationError("Invalid email format".to_string()));
    }
    Ok(())
}

pub fn process_order(mut order: Order) -> Result<Order> {
    if order.items.is_empty() {
        return Err(AppError::ValidationError("Order has no items".to_string()));
    }

    order.total = calculate_order_total(&order.items);
    order.status = OrderStatus::Processing;
    Ok(order)
}

// Async function (requires tokio)
#[cfg(feature = "async")]
pub async fn fetch_user(id: u64) -> Result<User> {
    // Simulated async fetch
    Ok(User {
        id,
        name: "Fetched User".to_string(),
        email: "user@example.com".to_string(),
        is_active: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_user() {
        let user = create_user(1, "Alice", "alice@example.com");
        assert_eq!(user.id, 1);
        assert_eq!(user.name, "Alice");
    }

    #[test]
    fn test_calculate_order_total() {
        let items = vec![
            OrderItem {
                product_id: "A".to_string(),
                quantity: 2,
                unit_price: 10.0,
            },
            OrderItem {
                product_id: "B".to_string(),
                quantity: 1,
                unit_price: 25.0,
            },
        ];
        assert_eq!(calculate_order_total(&items), 45.0);
    }

    #[test]
    fn test_validate_email() {
        assert!(validate_email("valid@email.com").is_ok());
        assert!(validate_email("invalid").is_err());
    }
}
