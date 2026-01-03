// Package sample provides example Go code for reproduction testing.
// Tests structs, interfaces, and Go idioms.
package sample

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// User represents a system user.
type User struct {
	ID        int       `json:"id"`
	Name      string    `json:"name"`
	Email     string    `json:"email"`
	IsActive  bool      `json:"is_active"`
	CreatedAt time.Time `json:"created_at"`
}

// Product represents an item in inventory.
type Product struct {
	SKU      string   `json:"sku"`
	Name     string   `json:"name"`
	Price    int64    `json:"price"`
	Quantity int      `json:"quantity"`
	Tags     []string `json:"tags"`
}

// Order represents a customer order.
type Order struct {
	ID     string      `json:"id"`
	UserID int         `json:"user_id"`
	Items  []OrderItem `json:"items"`
	Total  int64       `json:"total"`
	Status string      `json:"status"`
}

// OrderItem represents an item in an order.
type OrderItem struct {
	ProductID string `json:"product_id"`
	Quantity  int    `json:"quantity"`
	UnitPrice int64  `json:"unit_price"`
}

// Repository defines storage operations.
type Repository interface {
	Get(id string) (interface{}, error)
	Save(item interface{}) error
	Delete(id string) error
}

// UserService handles user operations.
type UserService struct {
	mu    sync.RWMutex
	users map[int]*User
}

// NewUserService creates a new UserService.
func NewUserService() *UserService {
	return &UserService{
		users: make(map[int]*User),
	}
}

// GetUser retrieves a user by ID.
func (s *UserService) GetUser(id int) (*User, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	user, ok := s.users[id]
	if !ok {
		return nil, errors.New("user not found")
	}
	return user, nil
}

// CreateUser creates a new user.
func (s *UserService) CreateUser(name, email string) (*User, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	id := len(s.users) + 1
	user := &User{
		ID:        id,
		Name:      name,
		Email:     email,
		IsActive:  true,
		CreatedAt: time.Now(),
	}
	s.users[id] = user
	return user, nil
}

// CalculateTotal calculates order total with tax.
func CalculateTotal(items []OrderItem, taxRate float64) int64 {
	var subtotal int64
	for _, item := range items {
		subtotal += int64(item.Quantity) * item.UnitPrice
	}
	return int64(float64(subtotal) * (1 + taxRate))
}

// FilterProducts filters products by predicate.
func FilterProducts(products []Product, predicate func(Product) bool) []Product {
	result := make([]Product, 0)
	for _, p := range products {
		if predicate(p) {
			result = append(result, p)
		}
	}
	return result
}

// FormatPrice formats price as currency string.
func FormatPrice(cents int64, currency string) string {
	dollars := float64(cents) / 100
	return fmt.Sprintf("%s %.2f", currency, dollars)
}
