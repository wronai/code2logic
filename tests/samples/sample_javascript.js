/**
 * Sample JavaScript file for reproduction testing.
 * Tests classes, functions, and async patterns.
 */

/**
 * User class representing a system user.
 */
class User {
    /**
     * Create a new User.
     * @param {number} id - User ID
     * @param {string} name - User name
     * @param {string} email - User email
     */
    constructor(id, name, email) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.isActive = true;
        this.createdAt = new Date();
    }

    /**
     * Get user display name.
     * @returns {string} Display name
     */
    getDisplayName() {
        return `${this.name} <${this.email}>`;
    }

    /**
     * Deactivate user account.
     */
    deactivate() {
        this.isActive = false;
    }
}

/**
 * Product class for inventory management.
 */
class Product {
    /**
     * Create a new Product.
     * @param {string} sku - Stock keeping unit
     * @param {string} name - Product name
     * @param {number} price - Price in cents
     */
    constructor(sku, name, price) {
        this.sku = sku;
        this.name = name;
        this.price = price;
        this.quantity = 0;
        this.tags = [];
    }

    /**
     * Add tags to product.
     * @param {...string} tags - Tags to add
     */
    addTags(...tags) {
        this.tags.push(...tags);
    }

    /**
     * Check if product is in stock.
     * @returns {boolean} True if in stock
     */
    isInStock() {
        return this.quantity > 0;
    }

    /**
     * Format price as currency.
     * @param {string} currency - Currency code
     * @returns {string} Formatted price
     */
    formatPrice(currency = 'USD') {
        const dollars = this.price / 100;
        return `${currency} ${dollars.toFixed(2)}`;
    }
}

/**
 * Calculate total with tax.
 * @param {number[]} items - Array of prices
 * @param {number} taxRate - Tax rate (default 0.1)
 * @returns {number} Total with tax
 */
function calculateTotal(items, taxRate = 0.1) {
    const subtotal = items.reduce((sum, item) => sum + item, 0);
    return subtotal * (1 + taxRate);
}

/**
 * Filter array by predicate.
 * @param {Array} arr - Array to filter
 * @param {Function} predicate - Filter function
 * @returns {Array} Filtered array
 */
function filterBy(arr, predicate) {
    return arr.filter(predicate);
}

/**
 * Fetch data from API.
 * @param {string} url - API URL
 * @returns {Promise<Object>} Response data
 */
async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Fetch failed:', error);
        return null;
    }
}

/**
 * Debounce function execution.
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in ms
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function (...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

/**
 * Deep clone an object.
 * @param {Object} obj - Object to clone
 * @returns {Object} Cloned object
 */
function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') {
        return obj;
    }
    if (Array.isArray(obj)) {
        return obj.map(item => deepClone(item));
    }
    const cloned = {};
    for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
            cloned[key] = deepClone(obj[key]);
        }
    }
    return cloned;
}

module.exports = {
    User,
    Product,
    calculateTotal,
    filterBy,
    fetchData,
    debounce,
    deepClone
};
