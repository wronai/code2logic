/**
 * Advanced JavaScript sample for testing TOON parser coverage.
 * Covers: IIFEs, nested functions, var declarations, function expressions,
 * module.exports, CommonJS require, arrow functions with params.
 */

const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');

// Arrow function assigned to const
const getArg = (name, def) => {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx !== -1) return process.argv[idx + 1];
  return def;
};

// Arrow function assigned to let
let processItem = (item) => {
  return item.trim().toLowerCase();
};

// Function expression assigned to const
const validate = function(input) {
  if (!input || typeof input !== 'string') return false;
  return input.length > 0;
};

// Async function expression assigned to const
const fetchResource = async function(url, options) {
  const response = await fetch(url, options);
  return response.json();
};

// var arrow function
var shouldIgnore = (filePath) => {
  return filePath.includes('node_modules');
};

// var function expression
var formatOutput = function(data, indent) {
  return JSON.stringify(data, null, indent || 2);
};

// Regular top-level function
function walk(dir, onFile) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) walk(full, onFile);
    else onFile(full);
  }
}

// Function with nested function inside
function findFiles(dir, extensions) {
  const files = [];

  function traverse(currentDir) {
    const items = fs.readdirSync(currentDir);
    for (const item of items) {
      const fullPath = path.join(currentDir, item);
      if (fs.statSync(fullPath).isDirectory()) {
        traverse(fullPath);
      } else if (extensions.includes(path.extname(item))) {
        files.push(fullPath);
      }
    }
  }

  traverse(dir);
  return files;
}

// Async top-level function
async function processFiles(pattern) {
  const files = findFiles('.', ['.js', '.ts']);
  for (const file of files) {
    const content = fs.readFileSync(file, 'utf8');
    console.log(`Processing: ${file}`);
  }
}

// Class with methods
class FileProcessor extends EventEmitter {
  constructor(rootDir) {
    super();
    this.rootDir = rootDir;
    this.results = [];
  }

  async analyze(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const result = { path: filePath, lines: content.split('\n').length };
    this.results.push(result);
    this.emit('analyzed', result);
    return result;
  }

  static fromConfig(configPath) {
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    return new FileProcessor(config.rootDir);
  }

  getResults() {
    return [...this.results];
  }
}

// Deeply nested function
function outerFunction(data) {
  function middleHelper(items) {
    function innerSort(a, b) {
      return a.localeCompare(b);
    }
    return items.sort(innerSort);
  }
  return middleHelper(data);
}

// IIFE with named function
(function main() {
  const root = getArg('root', '.');
  console.log(`Scanning ${root}...`);
  walk(root, (file) => {
    console.log(file);
  });
})();

// module.exports with shorthand properties
module.exports = {
  getArg,
  processItem,
  validate,
  fetchResource,
  shouldIgnore,
  formatOutput,
  walk,
  findFiles,
  processFiles,
  FileProcessor,
  outerFunction
};
