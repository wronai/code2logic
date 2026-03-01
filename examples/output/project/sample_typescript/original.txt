/**
 * Sample TypeScript file for reproduction testing.
 * Tests interfaces, generics, and type annotations.
 */

// Interfaces
interface User {
    id: number;
    name: string;
    email: string;
    isActive: boolean;
    createdAt: Date;
}

interface Product {
    sku: string;
    name: string;
    price: number;
    quantity: number;
    tags: string[];
}

interface OrderItem {
    productId: string;
    quantity: number;
    unitPrice: number;
}

interface Order {
    id: string;
    userId: number;
    items: OrderItem[];
    total: number;
    status: 'pending' | 'processing' | 'shipped' | 'delivered';
}

// Generic types
type Result<T, E = Error> = { success: true; data: T } | { success: false; error: E };

type Nullable<T> = T | null;

// Utility functions
function createUser(id: number, name: string, email: string): User {
    return {
        id,
        name,
        email,
        isActive: true,
        createdAt: new Date(),
    };
}

function calculateOrderTotal(items: OrderItem[]): number {
    return items.reduce((total, item) => total + item.quantity * item.unitPrice, 0);
}

function filterByStatus<T extends { status: string }>(items: T[], status: string): T[] {
    return items.filter(item => item.status === status);
}

function groupBy<T, K extends keyof T>(items: T[], key: K): Map<T[K], T[]> {
    const groups = new Map<T[K], T[]>();
    for (const item of items) {
        const groupKey = item[key];
        const group = groups.get(groupKey) || [];
        group.push(item);
        groups.set(groupKey, group);
    }
    return groups;
}

// Async functions
async function fetchUser(id: number): Promise<Result<User>> {
    try {
        const response = await fetch(`/api/users/${id}`);
        if (!response.ok) {
            return { success: false, error: new Error(`HTTP ${response.status}`) };
        }
        const data = await response.json();
        return { success: true, data };
    } catch (error) {
        return { success: false, error: error as Error };
    }
}

async function processOrder(order: Order): Promise<Result<Order>> {
    if (order.items.length === 0) {
        return { success: false, error: new Error('Order has no items') };
    }
    
    const total = calculateOrderTotal(order.items);
    const processedOrder: Order = {
        ...order,
        total,
        status: 'processing',
    };
    
    return { success: true, data: processedOrder };
}

// Class with generics
class Repository<T extends { id: number | string }> {
    private items: Map<T['id'], T> = new Map();

    add(item: T): void {
        this.items.set(item.id, item);
    }

    get(id: T['id']): Nullable<T> {
        return this.items.get(id) || null;
    }

    getAll(): T[] {
        return Array.from(this.items.values());
    }

    delete(id: T['id']): boolean {
        return this.items.delete(id);
    }

    count(): number {
        return this.items.size;
    }
}

// Export
export {
    User,
    Product,
    Order,
    OrderItem,
    Result,
    Nullable,
    createUser,
    calculateOrderTotal,
    filterByStatus,
    groupBy,
    fetchUser,
    processOrder,
    Repository,
};
