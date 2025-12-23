const { Octokit } = require("@octokit/action");
const fs = require('fs');
const path = require('path');
const core = require('@actions/core');

// 1. กำหนดค่าคงที่ (Marker ที่คุณกำหนดเอง)
const REPO_OWNER = 'ntwkkm';
const REPO_NAME = 'pl';
const README_PATH = 'README.md';

// Marker ที่คุณเลือก
const START_MARKER = '[--- REPOSITORY-TREE-START ---]'; // <--- แก้ไข
const END_MARKER = '[--- REPOSITORY-TREE-END ---]';   // <--- แก้ไข

const octokit = new Octokit();

/**
 * Retrieve the repository file tree from GitHub and return a filtered, sorted list of tree items.
 *
 * @returns {Array<Object>} An array of Git tree objects (files and folders) from the repository's main branch, filtered to exclude certain files/folders and sorted by path.
 * @throws {Error} If the GitHub API requests fail.
 */
async function getRepoTree() {
    try {
        // A. ดึง SHA ของ Branch (main)
        const { data: branch } = await octokit.request('GET /repos/{owner}/{repo}/branches/{branch}', {
            owner: REPO_OWNER,
            repo: REPO_NAME,
            branch: 'main'
        });
        const treeSha = branch.commit.commit.tree.sha;

        // B. ดึงโครงสร้าง Tree แบบ Recursive
        const { data: tree } = await octokit.request('GET /repos/{owner}/{repo}/git/trees/{tree_sha}?recursive=1', {
            owner: REPO_OWNER,
            repo: REPO_NAME,
            tree_sha: treeSha
        });

        // C. กรองและจัดเรียงรายการ
        const excludedPaths = ['README.md', 'index.html', 'style.css'];
        const excludedFolders = ['.github', 'node_modules', '.git'];

        const items = tree.tree
            .filter(item => {
                if (excludedPaths.includes(item.path) || item.path.startsWith('.')) {
                    return false;
                }
                if (excludedFolders.some(folder => item.path.startsWith(folder + '/'))) {
                    return false;
                }
                if (item.type === 'tree' && excludedFolders.includes(item.path)) {
                    return false;
                }
                return true;
            })
            .sort((a, b) => a.path.localeCompare(b.path));
            
        return items;

    } catch (error) {
        console.error('Error fetching repository tree:', error.message);
        throw error;
    }
}


/**
 * Build a Markdown block that shows the repository file tree for inclusion in a README.
 *
 * @param {Array<{path: string, type: string}>} items - Flat list of repository entries; each entry includes `path` and `type` ('blob' for files, 'tree' for folders').
 * @returns {string} Markdown containing a human-readable text tree wrapped in a fenced code block.
 */
function generateMarkdown(items) {
    let markdown = '📂 Repository Contents (File Structure)\n\n';
    markdown += 'This content reflects the repository structure (updated by GitHub Actions):\n\n';
    markdown += '```text\n'; // เริ่ม Code Block สำหรับ Tree Structure

    const rootItems = {}; // ใช้สำหรับสร้างโครงสร้าง Tree
    const processedFolders = new Set();

    // 1. จัดโครงสร้างเป็น Object/Tree
    items.forEach(item => {
        const parts = item.path.split('/');
        let currentLevel = rootItems;
        
        // สร้างโครงสร้างโฟลเดอร์ตาม Path
        for (let i = 0; i < parts.length - 1; i++) {
            const part = parts[i];
            if (!currentLevel[part]) {
                currentLevel[part] = { type: 'tree', children: {} };
            }
            currentLevel = currentLevel[part].children;
        }

        const lastPart = parts[parts.length - 1];
        if (item.type === 'blob') {
            currentLevel[lastPart] = { type: 'blob', path: item.path };
        } else if (item.type === 'tree') {
            // โฟลเดอร์ที่ยังไม่มีไฟล์ย่อย (กรณี Tree API ส่ง Tree ที่ไม่มีไฟล์ย่อยมา)
            if (!currentLevel[lastPart]) {
                currentLevel[lastPart] = { type: 'tree', children: {} };
            }
        }
    });

    /**
     * Render a nested tree node into the surrounding `markdown` string as a text-based tree view.
     *
     * Traverses the given node (a mapping of entry name → entry object) in sorted order and appends lines with ASCII connectors to the outer-scoped `markdown` variable. Folder entries (type `'tree'`) are rendered with a trailing `/` and recursed into using an updated `prefix`; file entries are rendered as leaf lines.
     *
     * @param {{ [name: string]: { type: 'tree' | 'blob', children?: object } }} node - Mapping of entry names to entry objects; folders use `type: 'tree'` and provide a `children` object, files use `type: 'blob'`.
     * @param {string} [prefix=''] - Current indentation and connector prefix applied to each line (used by recursion).
     */
    function traverse(node, prefix = '') {
        const keys = Object.keys(node).sort();
        
        keys.forEach((key, index) => {
            const isLast = index === keys.length - 1;
            const item = node[key];
            const connector = isLast ? '`-- ' : '|-- ';
            
            markdown += `${prefix}${connector}${key}`;
            
            if (item.type === 'tree') {
                markdown += ' /';
                markdown += '\n';
                // Recursive call สำหรับโฟลเดอร์ย่อย
                const newPrefix = prefix + (isLast ? '    ' : '|   ');
                traverse(item.children, newPrefix);
            } else {
                markdown += '\n';
            }
        });
    }

    // 3. เริ่มแสดงผลจาก Root
    traverse(rootItems);

    markdown += '```\n'; // สิ้นสุด Code Block
    return markdown;
}


/**
 * Replace the Markdown section in README.md delimited by START_MARKER and END_MARKER with the provided content.
 *
 * If the markers are missing or incorrectly ordered, the function logs an error and prepends the provided Markdown
 * to the existing README content (which may create duplicate sections).
 *
 * @param {string} newMarkdown - Markdown content to insert between START_MARKER and END_MARKER.
 */
async function updateReadme(newMarkdown) {
    const fullReadmePath = path.join(process.cwd(), README_PATH);
    let readmeContent;
    
    try {
        readmeContent = fs.readFileSync(fullReadmePath, 'utf8');
    } catch (e) {
        console.error(`Error reading ${README_PATH}:`, e);
        return;
    }

    const startIdx = readmeContent.indexOf(START_MARKER);
    const endIdx = readmeContent.indexOf(END_MARKER);

    // ตรวจสอบความถูกต้องของ Markers
    if (startIdx === -1 || endIdx === -1 || startIdx >= endIdx) {
        console.error(`ERROR: START_MARKER (${START_MARKER}) or END_MARKER (${END_MARKER}) not found/incorrectly placed in ${README_PATH}.`);
        console.log(`Ensure these markers are present: ${START_MARKER} and ${END_MARKER}`);
        // ถ้าหาไม่เจอ ให้ทำการ Prepend (ซึ่งเป็นสิ่งที่ทำให้เกิดเนื้อหาซ้ำ)
        const newContent = newMarkdown + '\n\n' + readmeContent;
        fs.writeFileSync(fullReadmePath, newContent, 'utf8');
        return;
    }

    // สร้างเนื้อหาใหม่: [ส่วนบน] + [START_MARKER + เนื้อหาใหม่ + END_MARKER] + [ส่วนล่าง]
    const before = readmeContent.substring(0, startIdx + START_MARKER.length);
    const after = readmeContent.substring(endIdx);
    
    // แทนที่เนื้อหาเก่าด้วยเนื้อหาใหม่
    const newContent = `${before}\n\n${newMarkdown}\n\n${after}`; // เพิ่มบรรทัดว่างเพื่อให้ดูดีขึ้น

    fs.writeFileSync(fullReadmePath, newContent, 'utf8');
    console.log(`${README_PATH} updated successfully by replacement.`);
}

/**
 * Orchestrates retrieval of the repository tree, generation of the Markdown representation, and update of the README.
 *
 * On error, logs the error message and marks the GitHub Action as failed.
 */
async function main() {
    try {
        const items = await getRepoTree();
        const newMarkdown = generateMarkdown(items);
        await updateReadme(newMarkdown);
    } catch (e) {
        console.error("Failed to run README generation:", e.message);
        core.setFailed(e.message);
    }
}

main();

// ต้องมี require statements สำหรับ GitHub Action Runner
// const core ถูก require ไว้ด้านบนแล้ว
// const github ถูก require ไว้ด้านบนแล้ว