<template>
  <div class="container">
    <div class="left-section">
      <!-- Input Section -->
      <div class="input-section">
        <el-input
          v-model="urlInput"
          placeholder="Enter URL to check"
          class="url-input"
          @keyup.enter="checkUrl"
        >
          <template #append>
            <el-button :loading="loading" @click="checkUrl" type="primary">
              Check URL
            </el-button>
          </template>
        </el-input>
        
        <!-- 新增文件上传组件 -->
        <el-upload
          class="upload-section"
          action="#"
          :auto-upload="false"
          :on-change="handleFileChange"
          :show-file-list="false"
          accept=".csv,.xlsx,.xls"
        >
          <el-button type="success" :loading="uploading" style="margin-top: 5px;">
            <el-icon><Upload /></el-icon>
            Upload Excel/CSV
          </el-button>
        </el-upload>

        <el-button 
      type="warning" 
      @click="exportToCSV"
      :loading="uploading"
      style="margin-top: 5px;"
    >
      <el-icon><Download /></el-icon>
      Export  CSV
    </el-button>
      </div>

      <!-- Statistics Cards -->
      <div class="stats-container">
        <el-card class="stat-card">
          <div class="stat-title">Total URLs</div>
          <div class="stat-value">{{ stats.total }}</div>
        </el-card>
        <el-card class="stat-card safe">
          <div class="stat-title">Safe Websites</div>
          <div class="stat-value">{{ stats.safe }}</div>
        </el-card>
        <el-card class="stat-card phishing">
          <div class="stat-title">Phishing Websites</div>
          <div class="stat-value">{{ stats.phishing }}</div>
        </el-card>
      </div>

      <!-- Charts -->
      <div class="chart-container">
        <v-chart class="chart" :option="chartOption" autoresize />
      </div>
    </div>

    <!-- History Table -->
    <div class="right-section">
      <div class="search-container">
      <el-input
        v-model="searchQuery"
        placeholder="Input URL"
        clearable
        @clear="handleSearchClear"
        @input="handleSearchInput"
        class="search-input"
      >
        <template #prefix>
          <el-icon><Search /></el-icon>
        </template>
      </el-input>
      <el-button-group style="margin-left: 10px;">
    <el-button 
      :type="filterStatus === null ? 'primary' : ''"
      @click="setFilterStatus(null)"
    >
      All
    </el-button>
    <el-button 
      :type="filterStatus === false ? 'legitimate' : ''"
      @click="setFilterStatus(false)"
    >
      legitimate
    </el-button>
    <el-button 
      :type="filterStatus === true ? 'phishing' : ''"
      @click="setFilterStatus(true)"
    >
      phishing
    </el-button>
  </el-button-group>
    </div>
      <el-table :data="currentPageData" class="history-table">
        <el-table-column prop="url" label="URL" width="400" align="center"/>
        <el-table-column label="Status" width="100">
          <template #default="scope">
            <el-tag 
              v-if="scope.row"
              :type="scope.row.isPhishing ? 'danger' : 'success'"
              effect="dark"
            >
              {{ scope.row.isPhishing ? 'Phishing' : 'Safe' }}
            </el-tag>
            <span v-else>N/A</span>
          </template>
        </el-table-column>
        <el-table-column prop="timestamp" label="Checked At" width="500"/>
      </el-table>
      <el-pagination
      class="pagination"
      @current-change="handleCurrentChange"
      :current-page="currentPage"
      :page-size="pageSize"
      layout="prev, pager, next, jumper"
      :total="filteredData.length">
    </el-pagination>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { ElInput, ElButton, ElTable, ElTableColumn, ElMessage, ElIcon, ElUpload,ElPagination } from 'element-plus'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { PieChart, BarChart } from 'echarts/charts'
import { Search, Upload, Download } from '@element-plus/icons-vue'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
} from 'echarts/components'
import VChart from 'vue-echarts'
import 'element-plus/dist/index.css'

use([
  CanvasRenderer,
  PieChart,
  BarChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
])

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001'

const urlInput = ref('')
const loading = ref(false)
const history = ref([])
const filterStatus = ref(null)

const stats = reactive({
  total: 0,
  safe: 0,
  phishing: 0
})

const chartOption = computed(() => ({
  title: {
    text: 'Website Safety Distribution',
    left: 'center'
  },
  tooltip: {
    trigger: 'item'
  },
  legend: {
    orient: 'vertical',
    left: 'left',
    top: '7%'
  },
  series: [
    {
      name: 'URL Status',
      type: 'pie',
      radius: '50%',
      data: [
        { value: stats.safe, name: 'Safe Websites' },
        { value: stats.phishing, name: 'Phishing Websites' }
      ],
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    }
  ]
}))

const currentPage = ref(1)
const pageSize = ref(10)

const currentPageData = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  const end = start + pageSize.value
  return filteredData.value.slice(start, end)
})

function handleCurrentChange(val) {
  currentPage.value = val
}
async function checkUrl() {
  if (!urlInput.value) {
    ElMessage.warning('Please enter a URL')
    return
  }

  try {
    loading.value = true
    // Replace with actual API call
    const response = await fetch(`${API_BASE_URL}/detect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: urlInput.value })
    })

    const result = await response.json()
    const newEntry = {
      url: urlInput.value,
      isPhishing: result.prediction === 'phishing',
      timestamp: new Date().toLocaleString()
    }

    history.value.unshift(newEntry)
    stats.total++
    result.prediction === 'phishing' ? stats.phishing++ : stats.safe++
    urlInput.value = ''
  } catch (error) {
    ElMessage.error('Error checking URL')
  } finally {
    loading.value = false
  }
}

async function uploadExcel(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch(`${API_BASE_URL}/upload_csv`, {
      method: 'POST',
      body: formData,
      timeout: 3000000,
    });
    
    if (!response.ok) {
      throw new Error('Upload failed');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error uploading file:', error);
    throw error;
  }
}

const uploading = ref(false)

// 处理文件上传
const handleFileChange = async (file) => {
  try {
    uploading.value = true
    const result = await uploadExcel(file.raw)
    
    // 确保result有数据且格式正确
    if (!result || !result.results || !Array.isArray(result.results)) {
      throw new Error('Invalid data format from server')
    }

    // 更新统计数据和历史记录
    const processedRecords = result.results
      .filter(record => record != null) // 过滤无效记录
      .map(record => ({
        url: record.original_url,
        isPhishing: Boolean(record.prediction === 'phishing'), // 确保是布尔值
        timestamp: new Date().toLocaleString()
      }))

    stats.total += result.total
    stats.safe += processedRecords.filter(r => !r.isPhishing).length
    stats.phishing += processedRecords.filter(r => r.isPhishing).length
    
    history.value.unshift(...processedRecords)
    
    ElMessage.success(`Successfully processed ${processedRecords.length} URL records
`)
  } catch (error) {
    ElMessage.error(error.message || 'failed to process file')
    console.error('Error processing file:', error)
  } finally {
    uploading.value = false
  }
}

const searchQuery = ref('')

const filteredData = computed(() => {
  let result = history.value
  
  // 始终执行URL搜索过滤
  const query = searchQuery.value.toLowerCase()
  result = result.filter(item => 
    item.url.toLowerCase().includes(query)
  )
  
  // 始终执行状态过滤
  if (filterStatus.value !== null) {
    result = result.filter(item => 
      item.isPhishing === filterStatus.value
    )
  }
  
  return result
})

function handleSearchInput() {
  // 搜索时重置到第一页
  currentPage.value = 1
}

function handleSearchClear() {
  searchQuery.value = ''
}


function setFilterStatus(status) {
  filterStatus.value = status
  currentPage.value = 1 // 重置到第一页
}

async function exportToCSV() {
  try {
    uploading.value = true
    const response = await fetch(`${API_BASE_URL}/export_csv`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        data: filteredData.value
      })
    })

    if (!response.ok) throw new Error('failed to export CSV')
    
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'phishing_results.csv'
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(url)
    document.body.removeChild(a)
    
    ElMessage.success('successfully exported to CSV')
  } catch (error) {
    ElMessage.error(error.message)
  } finally {
    uploading.value = false
  }
}
</script>

<style scoped>
.container {
  max-width: 2000px;
  margin: 0 auto;
  padding: 2rem;
  display: flex;
  gap: 2rem;
  background: #f5f7fa;
  min-height: 100vh;
}

.left-section {
  flex: 1;
  background: white;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.right-section {
  width: 1000px;
  background: white;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.input-section {
  margin-bottom: 2rem;
}

.url-input {
  max-width: 600px;
  margin: 0 auto;
}

.stats-container {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.stat-card {
  text-align: center;
  padding: 1.5rem;
  transition: all 0.3s;
  cursor: pointer;
}

.stat-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.stat-title {
  font-size: 1.1rem;
  color: #666;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.stat-value {
  font-size: 2.2rem;
  font-weight: bold;
  color: #333;
}

.safe {
  border-top: 4px solid #67c23a;
  background: rgba(103, 194, 58, 0.05);
}

.phishing {
  border-top: 4px solid #f56c6c;
  background: rgba(245, 108, 108, 0.05);
}

.chart-container {
  height: 400px;
  margin-bottom: 2rem;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  padding: 1rem;
}

.history-table {
  height: calc(100vh - 8rem);
  overflow-y: auto;
  border: 1px solid #ebeef5;
  border-radius: 4px;
}

.history-table :deep(.el-table__header) {
  background-color: #f8f8f9;
}

.history-table :deep(.el-table__row) {
  transition: all 0.2s;
  height: 50px;
}

.history-table :deep(.el-table__row:hover) {
  background-color: #f5f7fa !important;
}

.history-table :deep(.el-table__row--striped) {
  background-color: #fafafa;
}

.history-table :deep(.el-table__cell) {
  padding: 12px 0;
  border-right: 1px solid #ebeef5;
  text-align: center !important;
}

.history-table :deep(.el-table__inner-wrapper::before) {
  background-color: #ebeef5;
}

.history-table :deep(.el-tag) {
  font-weight: bold;
  padding: 0 8px;
  height: 24px;
  line-height: 24px;
}
.pagination {
  margin-top: 20px;
  justify-content: center;
  display: flex;
}

.right-section {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 4rem);
}

.history-table {
  flex: 1;
  overflow-y: auto;
  margin-bottom: 20px;
  height: calc(100vh - 4rem);
}
.search-container {
  margin-bottom: 20px;
}
.search-input {
  width: 400px;
}
</style>
